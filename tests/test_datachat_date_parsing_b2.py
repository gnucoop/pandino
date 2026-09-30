"""B2: column-level date interpretation in `trend` and `top_rows`.

No global day-first/month-first default: a day/month order is used only when the
column's values decide it, the whole column is read under one order, ambiguous
columns are refused, timezones are never converted, and `trend` discloses rows
whose date could not be interpreted.
"""

import datetime
import warnings

import pandas as pd
import pytest
from flask import Flask

from datachat.result_provenance import lookup_trusted_result
from datachat.tools.top_rows_tool import TopRowsTool
from datachat.tools.trend_tool import TrendTool

_app = Flask(__name__)


def _trend(df, **kwargs):
    kwargs.setdefault("date_col", "d")
    kwargs.setdefault("freq", "day")
    kwargs.setdefault("op", "count")
    with _app.test_request_context("/datachat"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no pandas parser warning may leak
            out = TrendTool(df).forward(**kwargs)
        return out, lookup_trusted_result(out)


def _top(df, **kwargs):
    kwargs.setdefault("sort_by", "d")
    kwargs.setdefault("n", 20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return TopRowsTool(df).forward(**kwargs)


def _periods(out):
    return [(r["period"], r["count"]) for r in out["data"]]


def _col(out, name="d"):
    return [r[name] for r in out["data"]]


def _df(values):
    return pd.DataFrame({"d": pd.Series(values, dtype=object)})


# ---------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------


def test_trend_infers_day_first_for_whole_column():
    out, facts = _trend(_df(["03/04/2026", "13/05/2026", "21/06/2026"]))
    assert _periods(out) == [("2026-04-03", 1), ("2026-05-13", 1), ("2026-06-21", 1)]
    assert "note" not in out and facts is None


def test_trend_infers_month_first_for_whole_column():
    out, _ = _trend(_df(["04/13/2026", "05/21/2026", "06/25/2026", "03/04/2026"]))
    assert _periods(out) == [("2026-03-04", 1), ("2026-04-13", 1), ("2026-05-21", 1), ("2026-06-25", 1)]


def test_trend_day_first_monthly_chronology():
    df = _df(["01/02/2026", "15/02/2026", "10/03/2026", "03/04/2026", "20/04/2026", "12/05/2026"])
    out, _ = _trend(df, freq="month")
    assert _periods(out) == [("2026-02", 2), ("2026-03", 1), ("2026-04", 2), ("2026-05", 1)]


@pytest.mark.parametrize(
    "values",
    [
        ["03/04/2026", "05/06/2026", "07/08/2026"],  # no value decides the order
        ["13/05/2026", "05/21/2026"],  # values contradict each other
    ],
)
def test_trend_refuses_ambiguous_day_month_order(values):
    out, _ = _trend(_df(values))
    assert out["kind"] == "error"
    assert out["code"] == "AMBIGUOUS_DATE_FORMAT"
    assert "data" not in out


def test_trend_same_day_and_month_is_not_ambiguous():
    out, _ = _trend(_df(["03/03/2026", "05/05/2026"]))
    assert _periods(out) == [("2026-03-03", 1), ("2026-05-05", 1)]


def test_trend_partial_population_is_disclosed():
    out, facts = _trend(_df(["13/05/2026", "invalid-date", None, "14/05/2026", ""]))
    assert out["kind"] == "table"
    assert _periods(out) == [("2026-05-13", 1), ("2026-05-14", 1)]
    note = "3 row(s) were excluded because their date value was missing or could not be interpreted."
    assert out["note"] == note
    assert facts is not None and facts.note == note


def test_trend_all_unparseable_keeps_explicit_error():
    out, _ = _trend(_df(["invalid-date", None, "nope"]))
    assert out["code"] == "NO_PARSEABLE_DATES"


def test_trend_numeric_column_is_not_a_date():
    out, _ = _trend(pd.DataFrame({"d": [1.5, 2.5, 3.0]}))
    assert out["code"] == "NO_PARSEABLE_DATES"


def test_trend_iso_dates_and_datetimes_still_work():
    out, facts = _trend(_df(["2026-05-13", "2026-05-13T10:30:00", "2026-01-02"]))
    assert _periods(out) == [("2026-01-02", 1), ("2026-05-13", 2)]
    assert facts is None


def test_trend_native_datetime_still_works():
    df = pd.DataFrame({"d": pd.to_datetime(["2026-05-13", "2026-01-02", None])})
    out, facts = _trend(df)
    assert _periods(out) == [("2026-01-02", 1), ("2026-05-13", 1)]
    assert facts.note.startswith("1 row(s) were excluded")


def test_trend_bounds_follow_column_day_first_policy():
    # Day-first column: 03/04 = 3 April, 04/03 = 4 March, 20/04 = 20 April.
    df = _df(["04/03/2026", "03/04/2026", "20/04/2026"])
    out, _ = _trend(df, start="01/04/2026", end="10/04/2026")
    # Month-first bounds (Jan 4 .. Oct 4) would have selected all three rows.
    assert _periods(out) == [("2026-04-03", 1)]


def test_trend_bounds_iso_on_day_first_column():
    df = _df(["04/03/2026", "03/04/2026", "20/04/2026"])
    out, _ = _trend(df, start="2026-04-01", end="2026-04-30")
    assert _periods(out) == [("2026-04-03", 1), ("2026-04-20", 1)]


def test_trend_ambiguous_bound_on_iso_column_is_refused():
    out, _ = _trend(_df(["2026-04-03", "2026-05-01"]), start="03/04/2026")
    assert out["code"] == "AMBIGUOUS_DATE_FORMAT"


def test_trend_invalid_bound_is_not_silently_ignored():
    out, _ = _trend(_df(["2026-04-03", "2026-05-01"]), start="not a date")
    assert out["code"] == "INVALID_DATE_BOUND"


# ---------------------------------------------------------------------------
# trend — timezones (never converted)
# ---------------------------------------------------------------------------


def test_trend_coherent_timezone_keeps_local_civil_date():
    # 23:30 in Rome is the next day in UTC: UTC normalization would move the bucket.
    df = pd.DataFrame({"d": pd.to_datetime(["2026-05-13 23:30", "2026-05-14 08:00"]).tz_localize("Europe/Rome")})
    out, _ = _trend(df, start="2026-05-13", end="2026-05-14")
    assert _periods(out) == [("2026-05-13", 1)]
    out, _ = _trend(df)
    assert _periods(out) == [("2026-05-13", 1), ("2026-05-14", 1)]


def test_trend_coherent_offset_strings():
    out, _ = _trend(_df(["2026-05-13T23:30:00+02:00", "2026-05-14T08:00:00+02:00"]))
    assert _periods(out) == [("2026-05-13", 1), ("2026-05-14", 1)]


@pytest.mark.parametrize(
    "values",
    [
        ["2026-05-13T10:00:00+02:00", "2026-05-14"],  # aware + naive
        ["2026-05-13T10:00:00+02:00", "2026-01-02T10:00:00+01:00"],  # different offsets
    ],
)
def test_trend_inconsistent_timezones_explicit_error(values):
    out, _ = _trend(_df(values))
    assert out["code"] == "INCONSISTENT_TIMEZONES"


def test_trend_aware_bound_on_naive_column_is_refused():
    out, _ = _trend(_df(["2026-05-13", "2026-05-14"]), start="2026-05-13T00:00:00+02:00")
    assert out["code"] == "INCONSISTENT_TIMEZONES"


# ---------------------------------------------------------------------------
# top_rows
# ---------------------------------------------------------------------------


def test_top_rows_day_first_sorts_chronologically():
    out = _top(_df(["03/04/2026", "04/03/2026", "12/05/2026", "13/05/2026"]), ascending=True)
    assert _col(out) == ["04/03/2026", "03/04/2026", "12/05/2026", "13/05/2026"]


def test_top_rows_month_first_sorts_chronologically():
    out = _top(_df(["04/03/2026", "03/04/2026", "05/13/2026"]), ascending=True)
    assert _col(out) == ["03/04/2026", "04/03/2026", "05/13/2026"]


def test_top_rows_refuses_ambiguous_dates():
    out = _top(_df(["03/04/2026", "05/06/2026", "07/08/2026"]))
    assert out["code"] == "AMBIGUOUS_DATE_FORMAT"
    assert "data" not in out


@pytest.mark.parametrize(
    ("ascending", "expected"),
    [(True, [0.3, 1.2, 1.5, 1.9]), (False, [1.9, 1.5, 1.2, 0.3])],
)
def test_top_rows_floats_sort_numerically(ascending, expected):
    out = _top(pd.DataFrame({"x": [1.2, 1.9, 1.5, 0.3]}), sort_by="x", ascending=ascending)
    assert _col(out, "x") == expected


def test_top_rows_mostly_text_is_not_date_sorted():
    # One parseable value must not turn the column temporal (previously "2026" sorted as a date).
    out = _top(_df(["apple", "2026-01-01", "banana"]), ascending=False)
    assert _col(out) == ["banana", "apple", "2026-01-01"]


def test_top_rows_unparseable_row_kept_last():
    df = pd.DataFrame({"d": ["14/05/2026", "nope", "13/05/2026", "02/05/2026"], "id": [1, 2, 3, 4]})
    assert _col(_top(df, ascending=True), "id") == [4, 3, 1, 2]
    out = _top(df, ascending=False)
    assert _col(out, "id") == [1, 3, 4, 2]
    assert _col(out) == ["14/05/2026", "13/05/2026", "02/05/2026", "nope"]  # source values unchanged
    assert "note" not in out


def test_top_rows_iso_and_native_still_work():
    assert _col(_top(_df(["2026-05-13", "2026-05-13T10:30:00", "2026-01-02"]))) == [
        "2026-05-13T10:30:00",
        "2026-05-13",
        "2026-01-02",
    ]
    df = pd.DataFrame({"d": pd.to_datetime(["2026-01-02", "2026-05-13"])})
    assert _col(_top(df)) == ["2026-05-13T00:00:00", "2026-01-02T00:00:00"]


def test_top_rows_coherent_timezone_sorts():
    df = pd.DataFrame({"d": pd.to_datetime(["2026-05-13 23:30", "2026-05-14 08:00"]).tz_localize("Europe/Rome")})
    assert _col(_top(df)) == ["2026-05-14T08:00:00+02:00", "2026-05-13T23:30:00+02:00"]


def test_top_rows_mixed_timezones_explicit_error():
    out = _top(_df(["2026-05-13T10:00:00+02:00", "2026-05-14", "2026-05-15"]))
    assert out["code"] == "INCONSISTENT_TIMEZONES"


# ---------------------------------------------------------------------------
# accepted formats: explicit full dates only, no free-form guessing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["April 03/04 2026", "May 2026", "2026-5", "3 May 2026", "Wed, 03 Apr 2026"])
def test_trend_free_form_and_partial_dates_are_not_interpreted(value):
    out, _ = _trend(_df([value, "2026-05-13"]))
    # Only the full ISO row is used; the free-form/partial value never becomes a date (no invented day 1).
    assert _periods(out) == [("2026-05-13", 1)]
    assert out["note"].startswith("1 row(s) were excluded")


def test_trend_free_form_only_column_has_no_parseable_dates():
    out, _ = _trend(_df(["April 03/04 2026", "May 2026", "2026-5"]))
    assert out["code"] == "NO_PARSEABLE_DATES"


def test_top_rows_free_form_values_are_not_date_sorted():
    # As dates this would be May 2026, April 2004, then zeta (NaT) last.
    out = _top(_df(["April 03/04 2026", "May 2026", "zeta"]), ascending=False)
    assert _col(out) == ["zeta", "May 2026", "April 03/04 2026"]


@pytest.mark.parametrize(
    ("value", "period"),
    [
        ("2026-05-13", "2026-05-13"),
        ("2026-5-13", "2026-05-13"),
        ("2026-05-13T10:30:00", "2026-05-13"),
        ("2026-05-13 10:30", "2026-05-13"),
        ("2026-05-13T10:30:00.123456", "2026-05-13"),
        ("2026-05-13T23:30:00+02:00", "2026-05-13"),
        ("2026-05-13T10:30:00Z", "2026-05-13"),
        ("2026/05/13", "2026-05-13"),
        ("2026.05.13", "2026-05-13"),
        ("13-05-2026", "2026-05-13"),
        ("13.05.2026 10:30", "2026-05-13"),
    ],
)
def test_trend_supported_full_date_formats(value, period):
    out, facts = _trend(_df([value]))
    assert _periods(out) == [(period, 1)]
    assert facts is None


def test_trend_leap_day_after_day_first_policy():
    out, _ = _trend(_df(["29/02/2024", "29/02/2026", "13/05/2026"]))
    assert _periods(out) == [("2024-02-29", 1), ("2026-05-13", 1)]
    assert out["note"].startswith("1 row(s) were excluded")


# ---------------------------------------------------------------------------
# timezones: DST and timezone-library mixes
# ---------------------------------------------------------------------------


def _rome_dst_df():
    # Europe/Rome switches to +02:00 on 2026-03-29.
    local = pd.to_datetime(["2026-03-28 23:30", "2026-03-29 23:30", "2026-03-30 00:30"])
    return pd.DataFrame({"d": local.tz_localize("Europe/Rome")})


def test_trend_native_named_timezone_across_dst():
    out, _ = _trend(_rome_dst_df())
    assert _periods(out) == [("2026-03-28", 1), ("2026-03-29", 1), ("2026-03-30", 1)]
    out, _ = _trend(_rome_dst_df(), start="2026-03-29", end="2026-03-29 23:59")
    assert _periods(out) == [("2026-03-29", 1)]


def test_top_rows_native_named_timezone_across_dst():
    assert _col(_top(_rome_dst_df())) == [
        "2026-03-30T00:30:00+02:00",
        "2026-03-29T23:30:00+02:00",
        "2026-03-28T23:30:00+01:00",
    ]


def test_mixed_timezone_libraries_explicit_error():
    from zoneinfo import ZoneInfo

    values = [
        pd.Timestamp("2026-01-13 10:00", tz="Europe/Rome"),  # pytz-backed
        pd.Timestamp(datetime.datetime(2026, 7, 13, 10, tzinfo=ZoneInfo("Europe/Rome"))),
    ]
    out, _ = _trend(_df(values))
    assert out["code"] == "INCONSISTENT_TIMEZONES"
    assert _top(_df(values))["code"] == "INCONSISTENT_TIMEZONES"


def test_top_rows_contradictory_day_month_refused():
    assert _top(_df(["13/05/2026", "05/21/2026", "01/02/2026"]))["code"] == "AMBIGUOUS_DATE_FORMAT"
