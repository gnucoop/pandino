import psycopg
import sys
from cryptography.fernet import Fernet, InvalidToken
import os
import base64
from datetime import datetime
from typing import Optional, Tuple
import logging
import pandas as pd 

# Generate a key for encryption and decryption
# Store the key in an environment variable or a secure file
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

KEY = os.environ.get("ENCRYPTION_KEY")
PGUSER = os.environ["PGUSER"]
PGPWD = os.environ["PGPWD"]
PGHOST = os.environ["PGHOST"]
PGDB = os.environ["PGDB"]

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


def connect():
    return psycopg.connect(host=PGHOST, dbname=PGDB, user=PGUSER, password=PGPWD)

def init_db():
    conn = connect()
    cursor = conn.cursor()
    sql_init =  """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            api_key TEXT NOT NULL UNIQUE,
            date_valid_until TEXT NOT NULL DEFAULT '2024-12-31',
            tokens INT NOT NULL DEFAULT 0
            CONSTRAINT tokens_nonnegative check (tokens >= 0)
        );
        CREATE TABLE IF NOT EXISTS logs (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            token_input INTEGER NOT NULL,
            token_output INTEGER NOT NULL,
            cost REAL NOT NULL,
            model TEXT NOT NULL,
            provider TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS costs (
            id SERIAL PRIMARY KEY,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            start_date_valid TEXT NOT NULL,
            end_date_valid TEXT NOT NULL,
            token_input_cost REAL NOT NULL,
            token_output_cost REAL NOT NULL
        );
    """
    # Execute the SQL script
    try:
        # Split the script into individual statements
        statements = sql_init.split(';')
        for statement in statements:
            if statement.strip():  # Skip empty statements
                cursor.execute(statement)
        conn.commit()
        print("Database initialized successfully.")
    except Exception as e:
        conn.rollback()
        print(f"An error occurred: {e}")
    finally:
        cursor.close()

def extend_expiration_date():
    current_date = datetime.now().strftime("%Y-%m-%d")
    new_date = pd.to_datetime(current_date)+pd.DateOffset(years= 1) 
    string_date = str(new_date)
    return string_date

def add_user(
    username: str,
    api_key: str,
    date_valid_until: Optional[str] = None
) -> Optional[str]:
    """
    Adds a new user to the 'users' table with an encrypted API key and optional expiration date.

    :param username: Unique username of the user.
    :param api_key: API key to be stored (will be encrypted before insertion).
    :param date_valid_until: Optional expiration date (ISO format). If not provided, 1 year is added.
    :return: None if success, or an error message string if an exception occurs.
    """
    if date_valid_until is None:
        date_valid_until = extend_expiration_date()

    logging.info(f"Adding user: {username} with expiration: {date_valid_until}")

    conn = connect()
    cursor = conn.cursor()
    encrypted_api_key = cipher_suite.encrypt(api_key.encode())

    try:
        cursor.execute(
            "INSERT INTO users (username, api_key, date_valid_until) VALUES (%s, %s, %s)",
            (username, encrypted_api_key.decode(), date_valid_until),
        )
        conn.commit()
        return None
    except psycopg.IntegrityError as e:
        logging.warning(f"IntegrityError while adding user {username}: {str(e)}")
        return f"Error adding new user: {e}"
    except Exception as e:
        logging.exception("Unexpected error in add_user")
        return f"Error adding new user: {e}"
    finally:
        conn.close()

    
def remove_user(username: str) -> Optional[str]:
    """
    Removes a user from the 'users' table by their username.

    :param username: The username of the user to be removed.
    :return: None if success, or an error message string if an exception occurs.
    """
    logging.info(f"Attempting to remove user: {username}")

    conn = connect()
    cursor = conn.cursor()

    query = "DELETE FROM users WHERE username = %s"

    try:
        cursor.execute(query, (username,))
        conn.commit()
        return None
    except psycopg.IntegrityError as e:
        logging.warning(f"IntegrityError while deleting user {username}: {str(e)}")
        return f"Error deleting user: {e}"
    except Exception as e:
        logging.exception("Unexpected error in remove_user")
        return f"Error deleting user: {e}"
    finally:
        conn.close()


def edit_tokens(username: str, tokens_quantity: int) -> tuple[bool, str]:
    """
    Updates the token balance for a user and extends their API key expiration date by one year.

    :param username: The username of the user to update.
    :param tokens_quantity: The number of tokens to add (can be negative).
    :return: Tuple (True, message) on success, or (False, error message) on failure.
    """
    date_valid_until = extend_expiration_date()
    logging.info(f"Editing tokens for user={username}, amount={tokens_quantity}")

    conn = connect()
    cursor = conn.cursor()

    query = """
        UPDATE users 
        SET tokens = tokens + %s, date_valid_until = %s 
        WHERE username = %s
    """

    try:
        cursor.execute(query, (tokens_quantity, date_valid_until, username))
        conn.commit()
        return True, "Tokens edited successfully"
    except psycopg.IntegrityError as e:
        logging.warning(f"IntegrityError while editing tokens for {username}: {str(e)}")
        return False, "Error while editing tokens"
    except Exception as e:
        logging.exception("Unexpected error in edit_tokens")
        return False, "Error while editing tokens"
    finally:
        conn.close()


def list_users():
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, api_key, date_valid_until, tokens FROM users")
    users = cursor.fetchall()
    conn.close()
    if users:
        print("Existing users:")
        for id, user, api_key, date_valid_until, tokens in users:
            try:
                decrypted_api_key = cipher_suite.decrypt(api_key).decode()
                print(
                    f"ID: {id}, Username: {user}, ApiKey: {decrypted_api_key}, Date Valid Until: {date_valid_until}, Tokens: {tokens}"
                )
            except InvalidToken:
                print(
                    f"Username: {user}, ApiKey: decryption failed, Date Valid Until: {date_valid_until}, Tokens: {tokens}"
                )
    else:
        print("No users found in the database.")


def get_user_by_username(user_name: str) -> Optional[dict[str, str | int]]:
    """
    Retrieves a user from the database by username and decrypts their API key.

    :param user_name: The username or email of the user to retrieve.
    :return: A dictionary containing user fields if found, or None if not found.
    """
    logging.info(f"Looking up user by username: {user_name}")

    conn = connect()
    cursor = conn.cursor()

    query = """
        SELECT id, username, api_key, date_valid_until, tokens 
        FROM users 
        WHERE username = %s
    """

    try:
        cursor.execute(query, (user_name,))
        user = cursor.fetchone()
    finally:
        conn.close()

    if user:
        try:
            decrypted_key = cipher_suite.decrypt(user[2]).decode("utf-8")
        except Exception as e:
            logging.error(f"Failed to decrypt API key for user {user_name}: {str(e)}")
            decrypted_key = "DECRYPTION_FAILED"

        user_data = {
            "id": user[0],
            "username": user[1],
            "api_key": decrypted_key,
            "date_valid_until": user[3],
            "tokens": user[4],
        }
        logging.info(f"User found: {user_data}")
        return user_data

    logging.warning(f"No user found for username: {user_name}")
    return None


def get_user_tokens(user_name: str) -> Optional[int]:
    """
    Retrieves the token count for a user by their username.

    :param user_name: The username of the user.
    :return: Number of tokens if user exists and the value is an int, otherwise None.
    """
    logging.info(f"Retrieving token count for user: {user_name}")

    user = get_user_by_username(user_name)
    if user is None:
        logging.warning(f"User not found: {user_name}")
        return None

    token_value = user["tokens"]
    return token_value if isinstance(token_value, int) else None



def validate_api_key(api_key: str, user_email: str) -> Tuple[bool, str]:
    """
    Validates whether the provided API key matches the user's stored (encrypted) key 
    and is still within the valid date range.

    :param api_key: The plain API key provided by the user.
    :param user_email: The username/email associated with the key.
    :return: Tuple (True, "match") if valid, otherwise (False, reason).
    """
    logging.info(f"Validating API key for user: {user_email}")

    conn = connect()
    cursor = conn.cursor()

    query = """
        SELECT api_key, date_valid_until 
        FROM users 
        WHERE username = %s
    """

    cursor.execute(query, (user_email,))
    encrypted_keys = cursor.fetchall()
    conn.close()

    if not encrypted_keys:
        return False, "No matching API key found"

    current_date = datetime.now().date()
    found_expired = False

    for encrypted_key, date_valid_until in encrypted_keys:
        try:
            expiration = datetime.strptime(date_valid_until, "%Y-%m-%d").date()
        except Exception as e:
            pass
        try:
            expiration = datetime.strptime(date_valid_until, "%Y-%m-%d %H:%M:%S").date()
        except Exception as e:
            logging.error(f"Invalid date format in DB for user {user_email}: {date_valid_until}")
            continue

        if expiration < current_date:
            found_expired = True
            continue

        try:
            decrypted_key = cipher_suite.decrypt(encrypted_key).decode().strip()
            if decrypted_key == api_key.strip():
                return True, "API key match found"
        except InvalidToken:
            continue
        except Exception:
            continue

    if found_expired:
        return False, "API key expired"

    return False, "No matching API key found"



def print_stored_keys():
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT username, api_key FROM users")
    users = cursor.fetchall()
    conn.close()
    print("Stored API keys:")
    for username, encrypted_key in users:
        print(f"Username: {username}, Encrypted key: {encrypted_key}")
        try:
            decrypted_key = cipher_suite.decrypt(encrypted_key).decode()
            print(f"  Decrypted key: {decrypted_key}")
        except Exception as e:
            print(f"  Error decrypting key: {str(e)}")


def log_token_usage(user_id, token_input, token_output, model, provider):
    conn = connect()
    cursor = conn.cursor()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Fetch the cost from the costs table
    current_date = datetime.now().strftime("%Y-%m-%d")
    cursor.execute(
        """
            SELECT token_input_cost, token_output_cost 
            FROM costs 
            WHERE provider = %s AND model = %s AND start_date_valid <= %s AND end_date_valid >= %s
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
            VALUES (%s, %s, %s, %s, %s, %s, %s)                                                                                                                                                                                                     
        """,
        (date, user_id, token_input, token_output, cost, model, provider),
    )
    conn.commit()
    conn.close()


def print_help():
    print("Usage: python database-pg.py <command>")
    print("Commands:")
    print("  init_db                     Initialize the database")
    print("  add_user <username> <api_key> <date_valid_until>  Add a new user")
    print("  remove_user <username> Removes an existing user")
    print("  get_user_by_username <user_name> Retrieve a user by its username/mail")
    print("  edit_tokens <user_name> <quantity> Adds or removes a user's tokens by the user's username/mail")
    print("  list_users                  List all users")
    print("  print_keys                  Print all stored API keys")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "init_db":
            init_db()
        elif sys.argv[1] == "add_user" and len(sys.argv) == 4:
            username, api_key = sys.argv[2], sys.argv[3]
            add_user(username, api_key)
        elif sys.argv[1] == "remove_user" and len(sys.argv) == 3:
            username = sys.argv[2]
            remove_user(username)
        elif sys.argv[1] == "get_user_by_username" and len(sys.argv) == 3:
            user_name = sys.argv[2]
            get_user_by_username(user_name)
        elif sys.argv[1] == "edit_tokens" and len(sys.argv) == 4:
            username, tokens_quantity = sys.argv[2], int(sys.argv[3])
            edit_tokens(username, tokens_quantity)
        elif sys.argv[1] == "list_users":
            list_users()
        elif sys.argv[1] == "print_keys":
            print_stored_keys()
        else:
            print_help()
    else:
        print_help()
