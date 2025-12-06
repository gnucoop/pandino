import psycopg
import sys
from cryptography.fernet import Fernet, InvalidToken
import os
import base64
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
import logging
import pandas as pd
import bcrypt

from database_methods import (
    build_get_user_by_username_query,
    build_add_user_query,
    build_remove_user_query,
    build_edit_tokens_query,
    build_list_users_query,
    build_print_stored_keys_query,
    build_validate_api_key_query,
    build_get_token_cost_query,
    build_insert_token_log_query,
    build_get_prompt_query,
    build_get_total_users_query,
    build_get_total_tokens_query,
    build_get_logs_for_admin_query,
    build_update_user_tokens_query,
    build_get_total_log_stats_query,
    build_get_daily_log_stats_query,
    build_get_top_users_by_token_usage_query,
    build_get_user_by_id_query,
    build_get_all_prompts_query,
    build_get_prompt_by_id_query,
    build_add_prompt_query,
    build_update_prompt_query,
    build_delete_prompt_query,
    build_add_cost_query,
    build_update_cost_query,
    build_delete_cost_query,
    build_get_all_costs_query,
    build_get_cost_by_id_query,
    build_get_daily_stats_query,
    build_get_recent_activity_query,
    build_query_associations_by_district_query,
    build_query_associations_by_product_query,
    build_query_product_in_district_query,
    build_query_associations_by_product_query,
    build_query_product_in_district_query,
    build_query_association_details_query,
    build_create_users_associazioni_table_query,
    build_get_farmer_by_username_query,
    build_get_association_by_id_query,
    build_update_association_query,
    build_get_products_by_association_query,
    build_add_product_query,
    build_update_product_query,
    build_delete_product_query,
    build_get_all_farmers_query
)

# Generate a key for encryption and decryption
# Store the key in an environment variable or a secure file
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

KEY = os.environ.get("ENCRYPTION_KEY")
PGUSER = os.environ["PGUSER"]
PGPWD = os.environ["PGPWD"]
PGHOST = os.environ["PGHOST"]
PGDB = os.environ["PGDB"]
PGPORT = os.getenv("PG_PORT", "5432")
schema = os.environ.get("MAUI_SCHEMA", "public")

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
    conn = psycopg.connect(host=PGHOST, dbname=PGDB, user=PGUSER, password=PGPWD, port=PGPORT)

    with conn.cursor() as cur:
        cur.execute(f"SET search_path TO {schema};")

    return conn


def init_db():
    conn = connect()
    cursor = conn.cursor()
    sql_init = """
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
        CREATE TABLE IF NOT EXISTS prompts (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            version INTEGER NOT NULL,
            message TEXT NOT NULL
        );
    """
    # Execute the SQL script
    try:
        # Split the script into individual statements
        statements = sql_init.split(";")
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
    new_date = pd.to_datetime(current_date) + pd.DateOffset(years=1)
    string_date = str(new_date)
    return string_date


def add_user(
    username: str, api_key: str, date_valid_until: Optional[str] = None
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
    encrypted_api_key = cipher_suite.encrypt(api_key.encode()).decode()

    try:
        query, params = build_add_user_query(
            username, encrypted_api_key, date_valid_until
        )
        cursor.execute(query, params)
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

    try:
        query, params = build_remove_user_query(username)
        cursor.execute(query, params)
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

    try:
        query, params = build_edit_tokens_query(
            tokens_quantity, date_valid_until, username
        )
        cursor.execute(query, params)
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
    """
    Retrieves and displays a list of users from the database,
    including decrypted API keys, expiration dates, and token balances.

    :return: None. Prints user information to the console.
    """
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_list_users_query()
        cursor.execute(query, params)
        users = cursor.fetchall()
    finally:
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

    query, params = build_get_user_by_username_query(user_name)

    try:
        cursor.execute(query, params)
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

    try:
        query, params = build_validate_api_key_query(user_email)
        cursor.execute(query, params)
        encrypted_keys = cursor.fetchall()
    finally:
        conn.close()

    if not encrypted_keys:
        return False, "No matching API key found"

    current_date = datetime.now().date()
    found_expired = False

    for encrypted_key, date_valid_until in encrypted_keys:
        try:
            expiration = datetime.strptime(date_valid_until, "%Y-%m-%d").date()
        except Exception:
            pass
        try:
            expiration = datetime.strptime(date_valid_until, "%Y-%m-%d %H:%M:%S").date()
        except Exception as e:
            logging.error(
                f"Invalid date format in DB for user {user_email}: {date_valid_until}"
            )
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


def print_stored_keys() -> None:
    """
    Prints all stored API keys from the database, including decrypted values if possible.

    Connects to the database, retrieves usernames and encrypted API keys, and attempts to decrypt each key.
    Handles decryption errors gracefully and displays relevant information for each user.

    :return: None
    """
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_print_stored_keys_query()
        cursor.execute(query, params)
        users = cursor.fetchall()
    finally:
        conn.close()

    print("Stored API keys:")
    for username, encrypted_key in users:
        print(f"Username: {username}, Encrypted key: {encrypted_key}")
        try:
            decrypted_key = cipher_suite.decrypt(encrypted_key).decode()
            print(f"  Decrypted key: {decrypted_key}")
        except Exception as e:
            print(f"  Error decrypting key: {str(e)}")


def log_token_usage(user_id, token_input, token_output, model, provider) -> None:
    """
    Logs token usage for a user by calculating the cost based on input and output tokens,
    and inserts a record into the token usage log in the database.

    :param user_id: The ID of the user whose token usage is being logged.
    :param token_input: Number of input tokens used.
    :param token_output: Number of output tokens generated.
    :param model: The model used for token processing.
    :param provider: The provider of the model.
    :return: None
    """
    conn = connect()
    cursor = conn.cursor()
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")

    # SELECT cost
    cost_query, cost_params = build_get_token_cost_query(provider, model, current_date)
    cursor.execute(cost_query, cost_params)
    cost_row = cursor.fetchone()
    if not cost_row:
        raise ValueError(f"Cost not found for provider: {provider} and model: {model}")

    token_input_cost, token_output_cost = cost_row
    cost = (token_input * token_input_cost) + (token_output * token_output_cost)

    # INSERT log
    insert_query, insert_params = build_insert_token_log_query(
        date_str, user_id, token_input, token_output, cost, model, provider
    )
    cursor.execute(insert_query, insert_params)
    conn.commit()
    conn.close()


def get_prompt_from_db(title: str, version: Optional[int] = None) -> Optional[str]:
    """
    Retrieves a prompt's message from the database by title, optionally filtering by version.

    :param title: The identifier of the prompt (e.g., 'start_chat_system').
    :param version: Optional specific version to retrieve. If None, retrieves the most recent version.
    :return: The prompt message as a string, or None if not found.
    """
    logging.info(f"Retrieving prompt from DB: title='{title}', version={version}")

    conn = connect()
    cursor = conn.cursor()

    query, params = build_get_prompt_query(title, version)

    try:
        cursor.execute(query, params)
        result = cursor.fetchone()
        if result:
            return result[0]  # message
        else:
            logging.warning(f"No prompt found in DB for title='{title}', version={version}")
            return None
    except Exception as e:
        logging.exception(f"Error retrieving prompt from DB: {str(e)}")
        return None
    finally:
        conn.close()

def get_users_for_admin():
    """
    Retrieves users from the database and returns them as a list of dictionaries
    for use in the admin panel.

    :return: List of user dictionaries with decrypted API keys
    """
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_list_users_query()
        cursor.execute(query, params)
        users_raw = cursor.fetchall()
    finally:
        conn.close()

    users = []
    if users_raw:
        for id, username, api_key, date_valid_until, tokens in users_raw:
            try:
                decrypted_api_key = cipher_suite.decrypt(api_key).decode()
            except InvalidToken:
                decrypted_api_key = "Decryption failed"
            
            is_active = False
            # Formatta la data se esiste
            if date_valid_until and hasattr(date_valid_until, 'strftime'):
                formatted_date = date_valid_until.strftime('%Y-%m-%d')
            else:
                formatted_date = str(date_valid_until) if date_valid_until else 'N/A'

            # Check if user is active (today < date_valid_until)
            if date_valid_until:
                try:
                    # Se è una stringa, prova a parsarla
                    if isinstance(date_valid_until, str):
                        # Prova vari formati
                        for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                            try:
                                valid_date = datetime.strptime(date_valid_until, fmt).date()
                                break
                            except ValueError:
                                continue
                        else:
                            # Se nessun formato funziona
                            valid_date = None
                        
                        if valid_date:
                            formatted_date = valid_date.strftime('%Y-%m-%d')
                    # Se è già un datetime
                    elif hasattr(date_valid_until, 'date'):
                        valid_date = date_valid_until.date()
                        formatted_date = valid_date.strftime('%Y-%m-%d')
                    # Se è già un date
                    elif hasattr(date_valid_until, 'year'):
                        valid_date = date_valid_until
                        formatted_date = valid_date.strftime('%Y-%m-%d')
                    else:
                        valid_date = None
                        formatted_date = str(date_valid_until)
                    
                    # Controlla se è attivo (oggi < data_scadenza)
                    if valid_date:
                        today = datetime.now().date()
                        is_active = today < valid_date
                        
                except (ValueError, AttributeError):
                    # Se il parsing fallisce, considera l'utente non attivo
                    is_active = False
                    formatted_date = str(date_valid_until)
            
            users.append({
                'id': id,
                'name': username,
                'email': decrypted_api_key,
                'tokens': f"{tokens} tokens",
                'is_active': is_active,
                'created_at': formatted_date
            })
    
    return users


def get_users_stats():
    """
    Get statistics about users for the admin dashboard.
    
    :return: Dictionary with user statistics
    """
    conn = connect()
    cursor = conn.cursor()
    
    try:
        # Count total users
        query, params = build_get_total_users_query()
        cursor.execute(query, params)
        total_users_row = cursor.fetchone()
        total_users = total_users_row[0] if total_users_row else 0
        
        # Sum total tokens
        query, params = build_get_total_tokens_query()
        cursor.execute(query, params)
        total_tokens_row = cursor.fetchone()
        total_tokens = total_tokens_row[0] if total_tokens_row else 0
        
    finally:
        conn.close()
    
    return {
        'total_users': total_users,
        'total_tokens': total_tokens
    }

def get_logs_for_admin(limit=100):
    """
    Retrieves logs from the database for the admin panel.
    
    :param limit: Maximum number of logs to retrieve
    :return: List of log dictionaries
    """
    conn = connect()
    cursor = conn.cursor()
    
    try:
        query, params = build_get_logs_for_admin_query(limit)
        cursor.execute(query, params)
        logs_raw = cursor.fetchall()
    finally:
        conn.close()
    
    logs = []
    if logs_raw:
        for log_id, user_id, username, date, token_input, token_output, cost, model, provider in logs_raw:
            # Formatta la data
            if date and hasattr(date, 'strftime'):
                formatted_date = date.strftime('%Y-%m-%d %H:%M:%S')
            else:
                formatted_date = str(date) if date else 'N/A'
            
            logs.append({
                'id': log_id,
                'user_id': user_id,
                'username': username or 'Unknown',
                'date': formatted_date,
                'token_input': token_input or 0,
                'token_output': token_output or 0,
                'cost': cost or 0,
                'model': model or 'N/A',
                'provider': provider or 'N/A'
            })
    
    return logs

def update_user_tokens(user_id, new_tokens):
    """
    Update the tokens field for a specific user.
    Sets date_valid_until to one year from today.
    
    :param user_id: ID of the user to update
    :param new_tokens: New token value
    :return: True if successful, False otherwise
    """
    conn = connect()
    cursor = conn.cursor()

    try:
        # Calculate date one year from today
        one_year_from_today = datetime.now().replace(microsecond=0) + timedelta(days=365)

        query, params = build_update_user_tokens_query(user_id, new_tokens, one_year_from_today)
        cursor.execute(query, params)
        conn.commit()
        
        if cursor.rowcount > 0:
            return True
        else:
            return False
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_logs_stats():
    """
    Get statistics about logs for charts and dashboard.
    
    :return: Dictionary with log statistics
    """
    conn = connect()
    cursor = conn.cursor()
    
    try:
        # Total tokens input/output
        query, params = build_get_total_log_stats_query()
        cursor.execute(query, params)
        totals = cursor.fetchone()
        
        # Tokens by day (last 7 days)
        query, params = build_get_daily_log_stats_query()
        cursor.execute(query, params)
        daily_stats = cursor.fetchall()
        
        # Top users by token usage
        query, params = build_get_top_users_by_token_usage_query()
        cursor.execute(query, params)
        top_users = cursor.fetchall()
        
    finally:
        conn.close()
    
    # Format daily stats
    daily_data = []
    if daily_stats:
        for day, input_t, output_t in daily_stats:
            day_str = day.strftime('%Y-%m-%d') if hasattr(day, 'strftime') else str(day)
            daily_data.append({
                'day': day_str,
                'input': input_t or 0,
                'output': output_t or 0
            })
    
    # Format top users
    top_users_data = []
    if top_users:
        for username, total in top_users:
            top_users_data.append({
                'username': username or 'Unknown',
                'total_tokens': total or 0
            })
    
    return {
        'total_input': totals[0] if totals and totals[0] is not None else 0,
        'total_output': totals[1] if totals and totals[1] is not None else 0,
        'total_cost': totals[2] if totals and totals[2] is not None else 0.0,
        'total_requests': totals[3] if totals and totals[3] is not None else 0,
        'daily_stats': daily_data,
        'top_users': top_users_data
    }

def get_user_by_id(user_id):
    """
    Get a single user by ID.
    
    :param user_id: ID of the user
    :return: User dictionary or None
    """
    conn = connect()
    cursor = conn.cursor()
    
    try:
        query, params = build_get_user_by_id_query(user_id)
        cursor.execute(query, params)
        user_data = cursor.fetchone()
        
        if user_data:
            id, username, api_key, date_valid_until, tokens = user_data
            
            try:
                decrypted_api_key = cipher_suite.decrypt(api_key).decode()
            except InvalidToken:
                decrypted_api_key = "Decryption failed"
            
            # Formatta la data se esiste
            if date_valid_until and hasattr(date_valid_until, 'strftime'):
                formatted_date = date_valid_until.strftime('%Y-%m-%d')
            else:
                formatted_date = str(date_valid_until) if date_valid_until else 'N/A'
            
            return {
                'id': id,
                'username': username,
                'api_key': decrypted_api_key,
                'date_valid_until': formatted_date,
                'tokens': tokens
            }
        return None
        
    finally:
        conn.close()

def get_all_prompts():
    """
    Retrieves all prompts from the database.

    :return: List of prompt dictionaries
    """
    conn = connect()
    cursor = conn.cursor()
    
    try:
        query, params = build_get_all_prompts_query()
        cursor.execute(query, params)
        prompts_raw = cursor.fetchall()
    finally:
        conn.close()
    
    prompts = []
    if prompts_raw:
        for id, title, version, message in prompts_raw:
            prompts.append({
                'id': id,
                'title': title,
                'version': version,
                'message': message,
            })
    
    return prompts

def get_prompt_by_id(prompt_id):
    """
    Get a single prompt by ID.
    
    :param prompt_id: ID of the prompt
    :return: Prompt dictionary or None
    """
    conn = connect()
    cursor = conn.cursor()
    
    try:
        query, params = build_get_prompt_by_id_query(prompt_id)
        cursor.execute(query, params)
        prompt_data = cursor.fetchone()
        
        if prompt_data:
            id, title, version, message = prompt_data
            return {
                'id': id,
                'title': title,
                'version': version,
                'message': message,
            }
        return None
        
    finally:
        conn.close()


def add_cost(
    model: str,
    provider: str,
    token_input_cost: float,
    token_output_cost: float,
    start_date_valid: str,
    end_date_valid: str,
) -> Optional[str]:
    """
    Adds a new cost entry to the 'costs' table.

    :param model: Model name.
    :param provider: Provider name.
    :param token_input_cost: Cost per input token.
    :param token_output_cost: Cost per output token.
    :param start_date_valid: Start date of validity.
    :param end_date_valid: End date of validity.
    :return: None if success, or an error message string if an exception occurs.
    """
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_add_cost_query(
            model,
            provider,
            token_input_cost,
            token_output_cost,
            start_date_valid,
            end_date_valid,
        )
        cursor.execute(query, params)
        conn.commit()
        return None
    except Exception as e:
        logging.exception("Unexpected error in add_cost")
        return f"Error adding new cost: {e}"
    finally:
        conn.close()


def update_cost(
    cost_id: int,
    model: str,
    provider: str,
    token_input_cost: float,
    token_output_cost: float,
    start_date_valid: str,
    end_date_valid: str,
) -> Optional[str]:
    """
    Updates a cost entry.

    :param cost_id: ID of the cost entry to update.
    :param model: New model name.
    :param provider: New provider name.
    :param token_input_cost: New input token cost.
    :param token_output_cost: New output token cost.
    :param start_date_valid: New start date.
    :param end_date_valid: New end date.
    :return: None if success, or an error message string if an exception occurs.
    """
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_update_cost_query(
            cost_id,
            model,
            provider,
            token_input_cost,
            token_output_cost,
            start_date_valid,
            end_date_valid,
        )
        cursor.execute(query, params)
        conn.commit()
        return None
    except Exception as e:
        logging.exception("Unexpected error in update_cost")
        return f"Error updating cost: {e}"
    finally:
        conn.close()


def delete_cost(cost_id: int) -> Optional[str]:
    """
    Deletes a cost entry from the 'costs' table by id.

    :param cost_id: The id of the cost entry to remove.
    :return: None if success, or an error message string if an exception occurs.
    """
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_delete_cost_query(cost_id)
        cursor.execute(query, params)
        conn.commit()
        return None
    except Exception as e:
        logging.exception("Unexpected error in delete_cost")
        return f"Error deleting cost: {e}"
    finally:
        conn.close()


def get_all_costs():
    """
    Retrieves all cost entries from the database.

    :return: List of cost dictionaries
    """
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_get_all_costs_query()
        cursor.execute(query, params)
        costs_raw = cursor.fetchall()
    finally:
        conn.close()

    costs = []
    if costs_raw:
        for id, model, provider, in_cost, out_cost, start, end in costs_raw:
            costs.append({
                'id': id,
                'model': model,
                'provider': provider,
                'token_input_cost': in_cost,
                'token_output_cost': out_cost,
                'start_date_valid': start,
                'end_date_valid': end,
            })

    return costs


def get_cost_by_id(cost_id: int):
    """
    Get a single cost entry by ID.

    :param cost_id: ID of the cost entry
    :return: Cost dictionary or None
    """
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_get_cost_by_id_query(cost_id)
        cursor.execute(query, params)
        cost_data = cursor.fetchone()

        if cost_data:
            id, model, provider, in_cost, out_cost, start, end = cost_data
            return {
                'id': id,
                'model': model,
                'provider': provider,
                'token_input_cost': in_cost,
                'token_output_cost': out_cost,
                'start_date_valid': start,
                'end_date_valid': end,
            }
        return None

    finally:
        conn.close()


def get_daily_stats(date: str):
    """
    Get total tokens and cost for a specific date.

    :param date: Date string in YYYY-MM-DD format.
    :return: Dictionary with daily stats
    """
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_get_daily_stats_query(date)
        cursor.execute(query, params)
        result = cursor.fetchone()
        
        total_tokens = result[0] if result and result[0] else 0
        total_cost = result[1] if result and result[1] else 0.0
        
        return {
            'total_tokens': total_tokens,
            'total_cost': total_cost
        }
    finally:
        conn.close()


def add_prompt(title: str, version: int, message: str) -> Optional[str]:
    """
    Adds a new prompt to the 'prompts' table.

    :param title: Title of the prompt.
    :param version: Version of the prompt.
    :param message: Message of the prompt.
    :return: None if success, or an error message string if an exception occurs.
    """
    logging.info(f"Adding prompt: {title}")

    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_add_prompt_query(
            title, version, message
        )
        cursor.execute(query, params)
        conn.commit()
        return None
    except psycopg.IntegrityError as e:
        logging.warning(f"IntegrityError while adding prompt {title}: {str(e)}")
        return f"Error adding new prompt: {e}"
    except Exception as e:
        logging.exception("Unexpected error in add_prompt")
        return f"Error adding new prompt: {e}"
    finally:
        conn.close()

def update_prompt(prompt_id: int, title: str, version: int, message: str) -> bool:
    """
    Update a prompt.
    
    :param prompt_id: ID of the prompt to update
    :param title: New title value.
    :param version: New version value.
    :param message: New message value.
    :return: True if successful, False otherwise
    """
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_update_prompt_query(prompt_id, title, version, message)
        cursor.execute(query, params)
        conn.commit()
        
        if cursor.rowcount > 0:
            return True
        else:
            return False
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def delete_prompt(prompt_id: int) -> bool:
    """
    Deletes a prompt from the 'prompts' table by its ID.

    :param prompt_id: The ID of the prompt to be removed.
    :return: True if deletion was successful, False otherwise.
    """
    logging.info(f"Attempting to remove prompt: {prompt_id}")

    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_delete_prompt_query(prompt_id)
        cursor.execute(query, params)
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logging.exception(f"Unexpected error in delete_prompt: {e}")
        return False
    finally:
        conn.close()


def get_recent_activity():
    """
    Get the 3 most recent activities (logs and new users).
    
    :return: List of activity dictionaries
    """
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_get_recent_activity_query()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        activities = []
        if rows:
            for type, date, details in rows:
                activities.append({
                    'type': type,
                    'date': date,
                    'details': details
                })
        return activities
    finally:
        conn.close()


def get_associations_by_district(district: str) -> List[Dict[str, Any]]:
    """
    Returns all associations located in a given district.
    Case-insensitive substring match via ILIKE is handled by the query builder.

    :param district: District name to filter (e.g., "Magude").
    :return: List of dictionaries representing associations.
    """

    from database_methods import build_query_associations_by_district_query

    query, params = build_query_associations_by_district_query(district)

    conn = connect()
    cursor = conn.cursor()

    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "nome_associazione": row[1],
                "soci_maschi": row[2],
                "soci_femmine": row[3],
                "descrizione": row[4],
                "distretto": row[5],
                "area_coltivata_ha": row[6],
                "sistema_irrigazione": row[7],
                "sistema_conservazione": row[8],
                "sistema_processamento": row[9],
            })

        return results

    except Exception as e:
        logging.exception("[database_pg] Error in get_associations_by_district")
        return []

    finally:
        conn.close()


def get_associations_by_product(product: str) -> List[Dict[str, Any]]:
    """
    Returns all associations that produce a given product.
    Case-insensitive substring match handled by the query builder.

    :param product: Product name (e.g., "milho", "feijao").
    :return: List of dictionaries with association + product info.
    """

    from database_methods import build_query_associations_by_product_query

    query, params = build_query_associations_by_product_query(product)

    conn = connect()
    cursor = conn.cursor()

    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row[0],                           # associazione.id
                "nome_associazione": row[1],
                "soci_maschi": row[2],
                "soci_femmine": row[3],
                "descrizione": row[4],
                "distretto": row[5],
                "area_coltivata_ha": row[6],
                "sistema_irrigazione": row[7],
                "sistema_conservazione": row[8],
                "sistema_processamento": row[9],
                "cultura": row[10],                     # prodotto.cultura
                "rendimento_estimado_kg": row[11],
                "preco_venda_estimado_kg": row[12],
            })

        return results

    except Exception as e:
        logging.exception("[database_pg] Error in get_associations_by_product")
        return []

    finally:
        conn.close()


def get_product_in_district(product: str, district: str) -> List[Dict[str, Any]]:
    """
    Returns associations that produce a given product AND are located in a given district.
    Case-insensitive substring match via ILIKE is handled by the query builder.

    :param product: Product name (e.g., "milho").
    :param district: District name (e.g., "Magude").
    :return: List of dictionaries with association + product info.
    """

    from database_methods import build_query_product_in_district_query

    query, params = build_query_product_in_district_query(product, district)

    conn = connect()
    cursor = conn.cursor()

    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "nome_associazione": row[1],
                "soci_maschi": row[2],
                "soci_femmine": row[3],
                "descrizione": row[4],
                "distretto": row[5],
                "area_coltivata_ha": row[6],
                "sistema_irrigazione": row[7],
                "sistema_conservazione": row[8],
                "sistema_processamento": row[9],
                "cultura": row[10],
                "rendimento_estimado_kg": row[11],
                "preco_venda_estimado_kg": row[12],
            })

        return results

    except Exception:
        logging.exception("[database_pg] Error in get_product_in_district")
        return []

    finally:
        conn.close()


def get_association_details(name: str) -> List[Dict[str, Any]]:
    """
    Returns detailed information about associations whose name matches
    the provided substring (case-insensitive).
    
    :param name: Partial or full association name (e.g., "Chipene", "chip").
    :return: List of dictionaries with full association details.
    """

    from database_methods import build_query_association_details_query

    query, params = build_query_association_details_query(name)

    conn = connect()
    cursor = conn.cursor()

    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "nome_associazione": row[1],
                "soci_maschi": row[2],
                "soci_femmine": row[3],
                "descrizione": row[4],
                "distretto": row[5],
                "area_coltivata_ha": row[6],
                "sistema_irrigazione": row[7],
                "sistema_conservazione": row[8],
                "sistema_processamento": row[9],
            })

        return results

    except Exception:
        logging.exception("[database_pg] Error in get_association_details")
        return []

    finally:
        conn.close()



# === Farmers Panel Wrappers ===

def verify_farmer_login(username: str, password_raw: str) -> Optional[dict]:
    """
    Verifies a farmer's login credentials using bcrypt.

    :param username: The username to check.
    :param password_raw: The raw password to check against the stored hash.
    :return: User dictionary if valid, None otherwise.
    """
    conn = connect()
    cursor = conn.cursor()
    
    try:
        query, params = build_get_farmer_by_username_query(username)
        cursor.execute(query, params)
        user = cursor.fetchone()
        
        if user:
            # user = (id, username, password_hash_db, id_associazione)
            stored_hash = user[2]
            
            # Check if stored_hash is a valid bcrypt hash
            try:
                # Ensure bytes
                pwd_bytes = password_raw.encode('utf-8')
                hash_bytes = stored_hash.encode('utf-8')
                
                if bcrypt.checkpw(pwd_bytes, hash_bytes):
                    return {
                        'id': user[0],
                        'username': user[1],
                        'id_associazione': user[3]
                    }
                else:
                    print(f"DEBUG: Password mismatch for {username}")
            except Exception as e:
                print(f"DEBUG: Error checking password for user {username}: {e}")
                # Fallback or just fail safely
                return None

    finally:
        conn.close()
    
    return None

def get_association_details(assoc_id) -> Optional[dict]:
    """
    Retrieves association details.

    :param assoc_id: ID of the association.
    :return: Association dictionary or None.
    """
    conn = connect()
    cursor = conn.cursor()
    
    try:
        query, params = build_get_association_by_id_query(assoc_id)
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        if row:
            # Mapping based on DB schema (from user request)
            # 1 id, 2 nome_associazione, 3 soci_maschi, 4 soci_femmine, 5 descrizione, 
            # 6 distretto, 7 area_coltivata_ha, 8 sistema_irrigazione, 9 sistema_conservazione, 
            # 10 sistema_processamento, 11 contatto_telefonico
            return {
                'id': row[0],
                'nome_associazione': row[1],
                'soci_maschi': row[2],
                'soci_femmine': row[3],
                'descrizione': row[4],
                'distretto': row[5],
                'area_coltivata_ha': row[6],
                'sistema_irrigazione': row[7],
                'sistema_conservazione': row[8],
                'sistema_processamento': row[9],
                'contatto_telefonico': row[10]
            }
    finally:
        conn.close()
        
    return None

def update_association_details(assoc_id, data) -> bool:
    """
    Updates association details.

    :param assoc_id: ID of the association.
    :param data: Dictionary of data to update.
    :return: True if successful, False otherwise.
    """
    conn = connect()
    cursor = conn.cursor()
    
    try:
        query, params = build_update_association_query(assoc_id, data)
        cursor.execute(query, params)
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error updating association {assoc_id}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_association_products(assoc_id) -> List[dict]:
    """
    Retrieves products for an association.

    :param assoc_id: ID of the association.
    :return: List of product dictionaries.
    """
    conn = connect()
    cursor = conn.cursor()
    
    products = []
    try:
        query, params = build_get_products_by_association_query(assoc_id)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        for row in rows:
            # 1 id, 2 cultura, 3 rendimento_estimado_kg, 4 preco_venda_estimado_kg, 5 id_associazione
            products.append({
                'id': row[0],
                'cultura': row[1],
                'rendimento_estimado_kg': row[2],
                'preco_venda_estimado_kg': row[3],
                'id_associazione': row[4]
            })
    finally:
        conn.close()
        
    return products

def add_association_product(assoc_id, data) -> bool:
    """
    Adds a product for an association.

    :param assoc_id: ID of the association.
    :param data: Dictionary of product data.
    :return: True if successful, False otherwise.
    """
    conn = connect()
    cursor = conn.cursor()
    
    # Ensure correct association ID
    data['id_associazione'] = assoc_id
    
    try:
        query, params = build_add_product_query(data)
        cursor.execute(query, params)
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error adding product for association {assoc_id}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def update_association_product(product_id, data) -> bool:
    """
    Updates a product.

    :param product_id: ID of the product.
    :param data: Dictionary of data to update.
    :return: True if successful, False otherwise.
    """
    conn = connect()
    cursor = conn.cursor()
    
    try:
        query, params = build_update_product_query(product_id, data)
        cursor.execute(query, params)
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error updating product {product_id}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def delete_association_product(product_id) -> bool:
    """
    Deletes a product.

    :param product_id: ID of the product.
    :return: True if successful, False otherwise.
    """
    conn = connect()
    cursor = conn.cursor()
    
    try:
        query, params = build_delete_product_query(product_id)
        cursor.execute(query, params)
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error deleting product {product_id}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()



def get_all_farmers_for_admin() -> List[dict]:
    """
    Retrieves all farmers with simplified association info for admin panel.

    :return: List of farmer dictionaries.
    """
    conn = connect()
    cursor = conn.cursor()
    
    farmers = []
    try:
        query, params = build_get_all_farmers_query()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        for row in rows:
            farmers.append({
                'id': row[0],
                'username': row[1],
                'id_associazione': row[2],
                'nome_associazione': row[3],
                'distretto': row[4]
            })
    finally:
        conn.close()
        
    return farmers

def print_help():
    print("Usage: python database-pg.py <command>")
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
