import base64
import io


# Converts a base64 string to a csv file
def base64toFile(b64):
    decrypted = base64.b64decode(b64).decode("utf-8")
    return io.StringIO(decrypted)
