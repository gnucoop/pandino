"""Column-level date interpretation shared by ``trend`` and ``top_rows``.

Policy:
- native datetime dtype is used as is; numeric and boolean dtypes are never dates;
- only full dates are accepted, through explicit patterns (no free-form
  parsing): ISO ``YYYY-MM-DD`` with optional ``[T ]HH:MM[:SS[.ffffff]]`` and
  optional ``Z``/``±HH:MM`` offset, and year-first ``YYYY/MM/DD`` / ``YYYY.MM.DD``;
  partial (year/month only) and month-name strings are not interpreted;
- numeric day/month strings (``03/04/2026``, ``03-04-2026``, ``03.04.2026``) are
  read under ONE order for the whole column, inferred only from values that
  rule the other order out (a component greater than 12). If no value decides
  (and some values differ between orders) or the values contradict each other,
  the column is ambiguous and is never guessed;
- timezones are never converted: one coherent timezone is kept; naive and
  aware values mixed, or several timezones/offsets, are refused.
"""

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

AMBIGUOUS_DATE_FORMAT = "AMBIGUOUS_DATE_FORMAT"
INCONSISTENT_TIMEZONES = "INCONSISTENT_TIMEZONES"
INVALID_DATE_BOUND = "INVALID_DATE_BOUND"

_TIME = r"(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?"
_DAY_MONTH_RE = re.compile(r"^(\d{1,2})([/.-])(\d{1,2})\2(\d{4})" + _TIME + r"$")
_YEAR_FIRST_RE = re.compile(r"^(\d{4})([/.])(\d{1,2})\2(\d{1,2})" + _TIME + r"$")
_ISO_RE = re.compile(
    r"^(\d{4})-(\d{1,2})-(\d{1,2})"
    r"(?:[T ](\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?(Z|[+-]\d{2}:?\d{2})?)?$"
)

_MISSING = object()
_BAD = object()


@dataclass(frozen=True)
class _DayMonth:
    first: int
    second: int
    year: int
    time: tuple[int, int, int]


@dataclass(frozen=True)
class ParsedDates:
    """Result of interpreting one column.

    ``values`` is a datetime series aligned with the source (NaT where a value is
    missing or not interpretable), or ``None`` when ``error`` is set.
    """

    values: Optional[pd.Series]
    error: Optional[str] = None
    present: int = 0
    date_like: int = 0
    dayfirst: Optional[bool] = None  # set only when the column decided the order


def date_error(code: str, column: str) -> dict[str, Any]:
    messages = {
        AMBIGUOUS_DATE_FORMAT: (
            f"Column '{column}' contains day/month dates whose order (dd/mm or mm/dd) "
            "cannot be determined unambiguously from the values."
        ),
        INCONSISTENT_TIMEZONES: f"Column '{column}' mixes date values with different or missing timezones.",
    }
    return {"kind": "error", "message": messages[code], "code": code}


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _time(groups: tuple[Optional[str], ...]) -> tuple[int, int, int]:
    h, mi, s = groups
    return (int(h or 0), int(mi or 0), int(s or 0))


def _build(
    year: int,
    month: int,
    day: int,
    time: tuple[int, int, int],
    microsecond: int = 0,
    tzinfo: Optional[_dt.tzinfo] = None,
) -> Any:
    try:
        return pd.Timestamp(
            _dt.datetime(year, month, day, time[0], time[1], time[2], microsecond, tzinfo=tzinfo)
        )
    except (ValueError, OverflowError):
        return _BAD


def _offset(txt: Optional[str]) -> Optional[_dt.tzinfo]:
    if not txt:
        return None
    if txt == "Z":
        return _dt.timezone.utc
    sign = -1 if txt[0] == "-" else 1
    digits = txt[1:].replace(":", "")
    try:
        return _dt.timezone(sign * _dt.timedelta(hours=int(digits[:2]), minutes=int(digits[2:])))
    except ValueError:
        return None


def _timestamp(value: Any) -> Any:
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        return _BAD
    return _BAD if pd.isna(ts) else ts


def _classify(v: Any) -> Any:
    if _is_missing(v):
        return _MISSING
    if isinstance(v, (bool, np.bool_)):
        return _BAD
    if isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date, np.datetime64)):
        return _timestamp(v)  # native value: no string parsing involved
    if not isinstance(v, str):
        return _BAD  # numbers are never dates
    txt = v.strip()
    if not txt:
        return _MISSING
    m = _DAY_MONTH_RE.match(txt)
    if m:
        return _DayMonth(int(m.group(1)), int(m.group(3)), int(m.group(4)), _time(m.groups()[4:7]))
    m = _YEAR_FIRST_RE.match(txt)
    if m:
        return _build(int(m.group(1)), int(m.group(3)), int(m.group(4)), _time(m.groups()[4:7]))
    m = _ISO_RE.match(txt)
    if m:
        tz_txt = m.group(8)
        tzinfo = _offset(tz_txt)
        if tz_txt and tzinfo is None:
            return _BAD
        fraction = m.group(7)
        microsecond = int(fraction.ljust(6, "0")) if fraction else 0
        return _build(int(m.group(1)), int(m.group(2)), int(m.group(3)), _time(m.groups()[3:6]), microsecond, tzinfo)
    return _BAD


def _resolve(item: _DayMonth, dayfirst: bool) -> Any:
    day, month = (item.first, item.second) if dayfirst else (item.second, item.first)
    return _build(item.year, month, day, item.time)


def _infer_dayfirst(items: list[_DayMonth]) -> tuple[Optional[bool], bool]:
    """Return (decided order or None, ambiguous?)."""
    day_evidence = any(i.first > 12 and i.second <= 12 for i in items)
    month_evidence = any(i.second > 12 and i.first <= 12 for i in items)
    if day_evidence and month_evidence:
        return None, True
    if day_evidence or month_evidence:
        return day_evidence, False
    orders_differ = any(i.first != i.second and i.first <= 12 and i.second <= 12 for i in items)
    return None, orders_differ


def _tz_key(ts: pd.Timestamp) -> Optional[str]:
    return None if ts.tz is None else str(ts.tz)


def parse_date_series(series: pd.Series) -> ParsedDates:
    present = int(series.notna().sum())
    if pd.api.types.is_datetime64_any_dtype(series):
        return ParsedDates(values=series, present=present, date_like=present)
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return ParsedDates(values=pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]"), present=present)

    items = [_classify(v) for v in series.tolist()]
    present = sum(1 for i in items if i is not _MISSING)
    date_like = sum(1 for i in items if i is not _MISSING and i is not _BAD)

    day_months = [i for i in items if isinstance(i, _DayMonth)]
    dayfirst, ambiguous = _infer_dayfirst(day_months)
    if ambiguous:
        return ParsedDates(values=None, error=AMBIGUOUS_DATE_FORMAT, present=present, date_like=date_like)
    order = True if dayfirst is None else dayfirst  # undecided only when every value reads the same both ways

    stamps = [_resolve(i, order) if isinstance(i, _DayMonth) else i for i in items]
    stamps = [s if isinstance(s, pd.Timestamp) else pd.NaT for s in stamps]
    if len({_tz_key(s) for s in stamps if s is not pd.NaT}) > 1:
        return ParsedDates(values=None, error=INCONSISTENT_TIMEZONES, present=present, date_like=date_like)

    try:
        index = pd.DatetimeIndex(stamps)
    except (ValueError, TypeError):  # e.g. one zone name from different tz libraries
        return ParsedDates(values=None, error=INCONSISTENT_TIMEZONES, present=present, date_like=date_like)
    values = pd.Series(index, index=series.index)
    return ParsedDates(values=values, present=present, date_like=date_like, dayfirst=dayfirst)


def parse_date_bound(value: Any, column: ParsedDates) -> tuple[Optional[pd.Timestamp], Optional[str]]:
    """Interpret a start/end filter value under the column's policy and timezone."""
    item = _classify(value)
    if item is _MISSING:
        return None, None
    if isinstance(item, _DayMonth):
        order = column.dayfirst
        if order is None:
            order, ambiguous = _infer_dayfirst([item])
            if ambiguous:
                return None, AMBIGUOUS_DATE_FORMAT
            order = True if order is None else order
        item = _resolve(item, order)
    if not isinstance(item, pd.Timestamp):
        return None, INVALID_DATE_BOUND

    col_tz = column.values.dt.tz if column.values is not None else None
    if col_tz is None:
        if item.tz is not None:
            return None, INCONSISTENT_TIMEZONES
        return item, None
    try:
        return (item.tz_localize(col_tz) if item.tz is None else item.tz_convert(col_tz)), None
    except Exception:  # nonexistent/ambiguous local time in the column's timezone
        return None, INVALID_DATE_BOUND
