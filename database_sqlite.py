import sqlite3
import sys
from cryptography.fernet import Fernet, InvalidToken
import os
import base64
import datetime

# Generate a key for encryption and decryption
# Store the key in an environment variable or a secure file
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

KEY = os.environ.get("ENCRYPTION_KEY")
if not KEY:
    print("Error: ENCRYPTION_KEY not found in environment variables.")
    sys.exit(1)

try:
    # Ensure the key is properly formatted
    KEY = base64.urlsafe_b64encode(
        base64.urlsafe_b64decode(KEY + "=" * (-len(KEY) % 4))
    )
    print("Using ENCRYPTION_KEY from environment variables.")
except Exception as e:
    print(f"Error with ENCRYPTION_KEY: {e}")
    sys.exit(1)

cipher_suite = Fernet(KEY)


def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    sql_init = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            api_key TEXT NOT NULL UNIQUE,
            date_valid_until TEXT NOT NULL DEFAULT '2024-12-31',
            tokens INT NOT NULL DEFAULT 0,
            CONSTRAINT tokens_nonnegative check (tokens >= 0)
        );
        -- Create the logs table
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            token_input INTEGER NOT NULL,
            token_output INTEGER NOT NULL,
            cost REAL NOT NULL,
            model TEXT NOT NULL,
            provider TEXT NOT NULL
        );
        -- Create the costs table
        CREATE TABLE IF NOT EXISTS costs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            start_date_valid TEXT NOT NULL,
            end_date_valid TEXT NOT NULL,
            token_input_cost REAL NOT NULL,
            token_output_cost REAL NOT NULL
        );
    """
    cursor.executescript(sql_init)
    conn.commit()
    conn.close()
    print("Database initialized successfully.")


def add_user(username, api_key, date_valid_until="2024-12-31"):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    encrypted_api_key = cipher_suite.encrypt(api_key.encode())
    try:
        cursor.execute(
            "INSERT INTO users (username, api_key, date_valid_until) VALUES (?, ?, ?)",
            (username, encrypted_api_key, date_valid_until),
        )
        conn.commit()
    except sqlite3.IntegrityError as e:
        return f"Error adding new user: {e}"
    except Exception as e:
        return f"Error adding new user: {e}"
    finally:
        conn.close()
        return None


def remove_user(username: str):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    try:
        query = f"DELETE FROM users WHERE username = '{username}'"
        cursor.execute(query)
        conn.commit()
    except sqlite3.IntegrityError as e:
        print(e)
        return f"Error deleting user: {e}"
    except Exception as e:
        print(e)
        return f"Error deleting user: {e}"
    finally:
        conn.close()
        return None


def edit_tokens(username, tokens_quantity):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE users SET tokens = tokens + ? WHERE username = ?",
            (tokens_quantity, username),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return False, "Error while editing Tokens"
    except Exception:
        return False, "Error while editing Tokens"
    finally:
        conn.close()
        return True, "Tokens edited successfully"


def list_users():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, api_key, date_valid_until, tokens FROM users")
    users = cursor.fetchall()
    conn.close()
    if users:
        print("Existing users:")
        for id, user, api_key, date_valid_until, tokens in users:
            print(
                f"ID: {id}, Username: {user}, ApiKey: {cipher_suite.decrypt(api_key.decode())}, Date Valid Until: {date_valid_until}, Tokens: {tokens}"
            )
    else:
        print("No users found in the database.")


def get_user_by_username(user_name: str) -> dict | None:
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = f"SELECT id, username, api_key, date_valid_until, tokens FROM users WHERE username='{user_name}'"
    cursor.execute(query)
    user = cursor.fetchone()
    conn.close()
    if user:
        userFound = {
            "id": user[0],
            "user": user[1],
            "api_key": cipher_suite.decrypt(user[2].decode()).decode("utf-8"),
            "date_valid_until": user[3],
            "tokens": user[4],
        }
        return userFound

    else:
        print("No users with this username found in the database.")
        return None


def get_user_tokens(user_name: str) -> int | None:
    user = get_user_by_username(user_name)
    if user is None:
        return None
    return user.get("tokens")


def validate_api_key(api_key, user_email):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT api_key, date_valid_until FROM users WHERE username='{user_email}'"
    )
    encrypted_keys = cursor.fetchall()
    conn.close()

    matchesLength = len(encrypted_keys)
    if not matchesLength:
        return False, "No matching API key found"
    from datetime import datetime

    current_date = datetime.now().strftime("%Y-%m-%d")
    for encrypted_key, date_valid_until in encrypted_keys:
        if date_valid_until < current_date:
            continue
        try:
            if isinstance(encrypted_key, str):
                encrypted_key = encrypted_key.encode()
            decrypted_key = cipher_suite.decrypt(encrypted_key).decode().strip()
            if decrypted_key == api_key.strip():
                return True, "API key match found"
        except InvalidToken:
            pass
        except Exception:
            pass
    if date_valid_until < current_date:
        return False, "API key expired"
    return False, "No matching API key found"


def print_stored_keys():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT username, api_key FROM users")
    users = cursor.fetchall()
    conn.close()
    print("Stored API keys:")
    for username, encrypted_key in users:
        print(f"Username: {username}, Encrypted key: {encrypted_key}")
        try:
            if isinstance(encrypted_key, str):
                encrypted_key = encrypted_key.encode()
            decrypted_key = cipher_suite.decrypt(encrypted_key).decode()
            print(f"  Decrypted key: {decrypted_key}")
        except Exception as e:
            print(f"  Error decrypting key: {str(e)}")


def log_token_usage(user_id, token_input, token_output, model, provider):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Fetch the cost from the costs table
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    cursor.execute(
        """
            SELECT token_input_cost, token_output_cost 
            FROM costs 
            WHERE provider = ? AND model = ? AND start_date_valid <= ? AND end_date_valid >= ?
        """,
        (provider, model, current_date, current_date),
    )
    cost_row = cursor.fetchone()
    if not cost_row:
        raise ValueError(f"Cost not found for provider: {provider} and model: {model}")

    token_input_cost, token_output_cost = cost_row
    cost = (token_input * token_input_cost) + (token_output * token_output_cost)

    cursor.execute(
        """                                                                                                                                                                                                                   
            INSERT INTO logs (date, user_id, token_input, token_output, cost, model, provider)                                                                                                                                           
            VALUES (?, ?, ?, ?, ?, ?, ?)                                                                                                                                                                                                     
        """,
        (date, user_id, token_input, token_output, cost, model, provider),
    )
    conn.commit()
    conn.close()


def print_help():
    print("Usage: python database_sqlite.py <command>")
    print("Commands:")
    print("  init_db                     Initialize the database")
    print("  add_user <username> <api_key> <date_valid_until>  Add a new user")
    print("  remove_user <username> Removes an existing user")
    print("  get_user_by_username <user_name> Retrieve a user by its username/mail")
    print(
        "  edit_tokens <user_name> <quantity> Adds or removes a user's tokens by the user's username/mail"
    )
    print("  list_users                  List all users")
    print("  print_keys                  Print all stored API keys")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "init_db":
            init_db()
        elif sys.argv[1] == "add_user" and len(sys.argv) == 5:
            username, api_key, date_valid_until = sys.argv[2], sys.argv[3], sys.argv[4]
            add_user(username, api_key, date_valid_until)
        elif sys.argv[1] == "remove_user" and len(sys.argv) == 3:
            username = sys.argv[2]
            remove_user(username)
        elif sys.argv[1] == "get_user_by_username" and len(sys.argv) == 3:
            user_name = sys.argv[2]
            get_user_by_username(user_name)
        elif sys.argv[1] == "edit_tokens" and len(sys.argv) == 4:
            username, tokens_quantity = sys.argv[2], sys.argv[3]
            edit_tokens(username, tokens_quantity)
        elif sys.argv[1] == "list_users":
            list_users()
        elif sys.argv[1] == "print_keys":
            print_stored_keys()
        else:
            print_help()
    else:
        print_help()