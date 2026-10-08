from typing import Any

INVALID_KEEP_COLUMNS = {
    "kind": "error",
    "code": "INVALID_KEEP_COLUMNS",
    "message": (
        "Invalid keep_columns: expected a list of distinct, existing column names, "
        "excluding the analyzed column and the result fields."
    ),
}


def validate_keep_columns(
    keep_columns: Any, column: str, reserved: tuple[str, ...]
) -> tuple[list[str], bool]:
    """
    The requested passthrough columns in caller order, and whether they are structurally valid.

    None means no passthrough columns. Names are taken exactly as given; existence in the
    source table is checked separately, once the table is known.
    """
    if keep_columns is None:
        return [], True
    if not isinstance(keep_columns, list):
        return [], False
    names: list[str] = []
    for name in keep_columns:
        if not isinstance(name, str) or not name.strip():
            return [], False
        if name in names or name == column or name in reserved:
            return [], False
        names.append(name)
    return names, True

