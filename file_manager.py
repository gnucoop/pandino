import base64
import io
import os

imageExtensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


# Converts a base64 string to a csv file
def base64toFile(b64):
    decrypted = base64.b64decode(b64).decode("utf-8")
    return io.StringIO(decrypted)


# Converts a filepath to a base64 string
def fileToBase64(filepath: str) -> str:
    with open(filepath, "rb") as file:
        convert = base64.b64encode(file.read())
        return str(convert)


# Checks if a text can be an image file path
def isImageFilePath(text: str) -> bool:
    if text.endswith(imageExtensions) and os.path.isfile(text):
        return True
    return False
