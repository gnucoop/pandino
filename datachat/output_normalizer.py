import os
import math
from typing import Any
import pandas as pd
from file_manager import fileToBase64, isImageFilePath


# Recursively replace NaN with None in dictionaries or lists.
def replace_nan(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: replace_nan(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_nan(item) for item in data]
    elif isinstance(data, float) and math.isnan(data):
        return None
    else:
        return data


def normalize_datachat_response(response: Any) -> dict[str, Any]:
    """
    Normalize PandasAI / pandas outputs into the exact response_dict structure
    currently returned by /datachat (AS-IS).
    """
    
    # --- Contract mode (new engines) ---
    if isinstance(response, dict) and "kind" in response:
        kind = str(response.get("kind") or "").strip().lower()

        if kind == "text":
            text = response.get("text", "")
            # Per restare AS-IS, trattiamo come stringa
            return {"type": "str", "value": str(text)}

        if kind == "error":
            message = response.get("message", "")
            return {"type": "str", "value": str(message)}

        if kind == "image_path":
            path = response.get("path")
            if isinstance(path, str) and isImageFilePath(path):
                return {"type": "image", "value": fileToBase64(path)}
            # fallback conservativo
            return {"type": "str", "value": str(path)}

        if kind == "table":
            data = response.get("data")
            # accettiamo DataFrame o records
            if isinstance(data, pd.DataFrame):
                return {
                    "type": "dataframe",
                    "value": replace_nan(data.to_dict(orient="records")),
                }
            if isinstance(data, list):
                try:
                    df = pd.DataFrame(data)
                    return {
                        "type": "dataframe",
                        "value": replace_nan(df.to_dict(orient="records")),
                    }
                except Exception as e:
                    raise RuntimeError(f"Failed to convert list to DataFrame: {str(e)}") from e

            if isinstance(data, dict):
                # caso raro: tabella già serializzata
                return {"type": "dict", **replace_nan(data)}

            return {"type": "str", "value": str(data)}

        # kind sconosciuto: fallback conservativo
        return {"type": "dict", **replace_nan(response)}


    # --- (old version) ---

    # 1) list -> DataFrame (tentativo)
    if isinstance(response, list):
        try:
            response = pd.DataFrame(response)
        except Exception as e:
            # IMPORTANT: qui NON possiamo jsonify né cambiare status code;
            # in AS-IS l'eccezione viene gestita nell'endpoint.
            # Quindi: rilanciamo la stessa eccezione così l'endpoint può comportarsi uguale.
            raise RuntimeError(f"Failed to convert list to DataFrame: {str(e)}") from e

    # 2) DataFrame -> dataframe records
    if isinstance(response, pd.DataFrame):
        return {
            "type": "dataframe",
            "value": replace_nan(response.to_dict(orient="records")),
        }

    # 3) dict -> dict + type
    if isinstance(response, dict):
        response_dict = replace_nan(response)
        if isinstance(response_dict, dict):
            response_dict.update({"type": "dict"})
            return response_dict
        # caso patologico: replace_nan ha trasformato in non-dict (non dovrebbe)
        return {"type": "dict", "value": response_dict}

    # 4) fallback -> string
    response_dict: dict[str, Any] = {"type": type(response).__name__, "value": str(response)}

    # Ricalco la logica AS-IS (inclusi i rami "strani") senza correggerla ora.
    if response_dict and response_dict.get("value"):
        if response_dict.get("type") == "string" and "plot" in response_dict:
            plot_path = response_dict.get("plot")
            if plot_path and os.path.exists(plot_path):
                response_dict["type"] = "text_and_image"
                response_dict["image"] = fileToBase64(plot_path)
                del response_dict["plot"]
        elif isinstance(response_dict.get("value"), str) and isImageFilePath(response_dict["value"]):
            response_dict["type"] = "image"
            response_dict["value"] = fileToBase64(response_dict["value"])
            
    return response_dict
