
import io
from typing import Any

import pandas as pd


def load_csv_to_dataframe(file_storage: Any) -> pd.DataFrame:
    """
    Load a CSV uploaded via Flask (request.files['file']) into a pandas DataFrame.
    AS-IS behavior: utf-8 decode, comma separator.
    """
    raw = file_storage.stream.read().decode("utf-8")
    return pd.read_csv(io.StringIO(raw), sep=",")
