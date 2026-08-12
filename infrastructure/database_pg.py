import psycopg
from psycopg import sql
import sys
from cryptography.fernet import Fernet, InvalidToken
import os
import base64
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Tuple, Dict, Any
import logging
import pandas as pd
from dotenv import load_dotenv

from config import AppConfig, load_config
from infrastructure.database_methods import (
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
    build_insert_feedback_query,
    build_get_feedback_for_admin_query,
    build_get_feedback_stats_query,
    build_get_feedback_sources_query,
    build_get_feedback_model_stats_query,
    build_get_total_feedback_count_query,
    build_get_total_logs_count_query,
    build_get_total_users_count_query,
    build_check_pgvector_maui_id_exists_query,
    build_check_table_exists_query,
    build_check_column_exists_query,
    build_add_column_query,
    build_insert_rag_file_query,
    build_get_all_rag_files_query,
    build_get_rag_file_for_delete_query,
    build_delete_rag_file_query,
    build_delete_pgvector_by_file_id_query,
)

logger = logging.getLogger(__name__)

KEY: Optional[bytes] = None
PGUSER: Optional[str] = None
PGPWD: Optional[str] = None
PGHOST: Optional[str] = None
PGDB: Optional[str] = None
PGPORT: Optional[str] = None
schema: Optional[str] = None
cipher_suite: Optional[Fernet] = None


def init(config: AppConfig) -> None:
    """Initialise module-level globals from AppConfig. Must be called once at startup."""
    global KEY, PGUSER, PGPWD, PGHOST, PGDB, PGPORT, schema, cipher_suite
    raw_key = config.encryption_key
    if not raw_key:
        raise ValueError("ENCRYPTION_KEY is required but not set.")
    try:
        KEY = base64.urlsafe_b64encode(
            base64.urlsafe_b64decode(raw_key + "=" * (-len(raw_key) % 4))
        )
    except Exception as e:
        raise ValueError(f"Invalid ENCRYPTION_KEY: {e}")
    cipher_suite = Fernet(KEY)
    PGUSER = config.database.user
    PGPWD = config.database.password
    PGHOST = config.database.host
    PGDB = config.database.db
    PGPORT = config.database.port
    schema = config.database.schema


def get_cipher_suite() -> Fernet:
    """Return the initialised Fernet instance or raise RuntimeError if init() was not called."""
    if cipher_suite is None:
        raise RuntimeError("database_pg.init() must be called before use.")
    return cipher_suite


def connect():
    if PGHOST is None or schema is None:
        raise RuntimeError("database_pg.init() must be called before connect().")
    conn = psycopg.connect(
        host=PGHOST, dbname=PGDB, user=PGUSER, password=PGPWD, port=PGPORT
    )
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
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
            provider TEXT NOT NULL,
            service TEXT,
            request_id TEXT
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
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            user_email TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            feedback_value TEXT NOT NULL CHECK (feedback_value IN ('positive', 'negative')),
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            log_id INTEGER REFERENCES logs(id) ON DELETE SET NULL,
            source TEXT
        );
        CREATE TABLE IF NOT EXISTS rag_files (
            id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            namespace TEXT NOT NULL,
            chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0),
            language TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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

    logger.info("event=user_add_started username=%s expires=%s", username, date_valid_until)

    conn = connect()
    cursor = conn.cursor()
    encrypted_api_key = get_cipher_suite().encrypt(api_key.encode()).decode()

    try:
        query, params = build_add_user_query(
            username, encrypted_api_key, date_valid_until
        )
        cursor.execute(query, params)
        conn.commit()
        return None
    except psycopg.IntegrityError as e:
        logger.warning("event=user_add_conflict username=%s error=%s", username, str(e))
        return f"Error adding new user: {e}"
    except Exception as e:
        logger.exception("event=user_add_failed")
        return f"Error adding new user: {e}"
    finally:
        conn.close()


def remove_user(username: str) -> Optional[str]:
    """
    Removes a user from the 'users' table by their username.

    :param username: The username of the user to be removed.
    :return: None if success, or an error message string if an exception occurs.
    """
    logger.info("event=user_remove_started username=%s", username)

    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_remove_user_query(username)
        cursor.execute(query, params)
        conn.commit()
        return None
    except psycopg.IntegrityError as e:
        logger.warning("event=user_remove_conflict username=%s error=%s", username, str(e))
        return f"Error deleting user: {e}"
    except Exception as e:
        logger.exception("event=user_remove_failed")
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
    logger.info("event=user_tokens_edit_started username=%s amount=%s", username, tokens_quantity)

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
        logger.warning("event=user_tokens_edit_conflict username=%s error=%s", username, str(e))
        return False, "Error while editing tokens"
    except Exception as e:
        logger.exception("event=user_tokens_edit_failed")
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
                decrypted_api_key = get_cipher_suite().decrypt(api_key).decode()
                print(
                    f"ID: {id}, Username: {user}, Date Valid Until: {date_valid_until}, Tokens: {tokens}"
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
    logger.info("event=user_lookup_started username=%s", user_name)

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
            decrypted_key = get_cipher_suite().decrypt(user[2]).decode("utf-8")
        except Exception as e:
            logger.error("event=user_lookup_key_decrypt_failed username=%s error=%s", user_name, str(e))
            decrypted_key = "DECRYPTION_FAILED"

        user_data = {
            "id": user[0],
            "username": user[1],
            "api_key": decrypted_key,
            "date_valid_until": user[3],
            "tokens": user[4],
        }
        logger.info("event=user_lookup_success username=%s", user_name)
        return user_data

    logger.warning("event=user_lookup_not_found username=%s", user_name)
    return None


def get_user_tokens(user_name: str) -> Optional[int]:
    """
    Retrieves the token count for a user by their username.

    :param user_name: The username of the user.
    :return: Number of tokens if user exists and the value is an int, otherwise None.
    """
    logger.info("event=user_tokens_lookup_started username=%s", user_name)

    user = get_user_by_username(user_name)
    if user is None:
        logger.warning("event=user_tokens_lookup_not_found username=%s", user_name)
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
    logger.info("event=api_key_validate_started username=%s", user_email)

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
            logger.error(
                "event=api_key_validate_date_invalid username=%s date=%s",
                user_email,
                date_valid_until,
            )
            continue

        if expiration < current_date:
            found_expired = True
            continue

        try:
            decrypted_key = get_cipher_suite().decrypt(encrypted_key).decode().strip()
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
        print(f"Username: {username}, Key stored: yes")
        try:
            get_cipher_suite().decrypt(encrypted_key).decode()
            print("  Decrypted key: available")
        except Exception as e:
            print(f"  Error decrypting key: {str(e)}")


def log_token_usage(
    user_id: int,
    token_input: int,
    token_output: int,
    model: str,
    provider: str,
    service: str,
    request_id: str,
) -> int:
    """
    Logs token usage for a user by calculating the cost based on input and output tokens,
    inserts a record into the 'logs' table, and returns the generated log ID.

    :param user_id: The ID of the user whose token usage is being logged.
    :param token_input: Number of input tokens used.
    :param token_output: Number of output tokens generated.
    :param model: The model used for token processing.
    :param provider: The provider of the model.
    :param service: The HTTP endpoint that produced this usage.
    :param request_id: The canonical HTTP request id of the request that produced this usage.
    :return: The ID of the inserted log record.
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

    # INSERT log (RETURNING id)
    insert_query, insert_params = build_insert_token_log_query(
        date_str,
        user_id,
        token_input,
        token_output,
        cost,
        model,
        provider,
        service,
        request_id,
    )
    cursor.execute(insert_query, insert_params)

    log_id_row = cursor.fetchone()
    if not log_id_row:
        raise RuntimeError("Failed to retrieve log_id after insert")

    log_id = log_id_row[0]

    conn.commit()
    conn.close()

    return log_id


def insert_rag_file(
    file_id: str,
    file_name: str,
    namespace: str,
    chunk_count: int,
    language: Optional[str],
) -> bool:
    """
    Inserts a new record into the rag_files table.

    :param file_id: Unique identifier of the document (file-level, hash-based).
    :param file_name: Original name of the uploaded file.
    :param namespace: Namespace (table) where embeddings are stored.
    :param chunk_count: Number of chunks generated from the document.
    :param language: Language of the document.
    :return: True if the insert succeeds, False otherwise.
    """
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_insert_rag_file_query(
            file_id=file_id,
            file_name=file_name,
            namespace=namespace,
            chunk_count=chunk_count,
            language=language,
        )
        cursor.execute(query, params)
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        logger.exception("event=rag_file_insert_failed")
        return False
    finally:
        conn.close()


def get_all_rag_files() -> list:
    """
    Retrieves all records from the rag_files table.

    :return: List of dictionaries with RAG file info.
    """
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_get_all_rag_files_query()
        cursor.execute(query, params)
        rows = cursor.fetchall()
    finally:
        conn.close()

    rag_files = []
    for file_id, file_name, namespace, chunk_count, language, created_at in rows:
        if created_at and hasattr(created_at, "strftime"):
            formatted_date = created_at.strftime("%Y-%m-%d %H:%M")
        else:
            formatted_date = str(created_at) if created_at else "N/A"

        rag_files.append(
            {
                "id": file_id,
                "file_name": file_name,
                "namespace": namespace,
                "chunk_count": chunk_count,
                "language": language or "-",
                "created_at": formatted_date,
            }
        )

    return rag_files


def delete_rag_file(file_id: str, namespace: str) -> dict:
    """
    Deletes a RAG file: its tracking row in rag_files and all of its vector
    chunks in the per-namespace PGVector table.

    Both deletes run in a single transaction on the same connection. The chunk
    table is only touched if it still exists (a namespace table may have been
    dropped independently), so a stale tracking row can always be cleaned up.

    :param file_id: File-level id (rag_files.id and chunk metadata 'file_id').
    :param namespace: Namespace of the file (already normalized when stored).
    :return: {"row_deleted": bool, "chunks_deleted": int}.
    """
    # Keep this in sync with vector_store.normalize_table_name without importing
    # it here, because vector_store imports this module.
    table_name = namespace.strip().lower().replace("-", "_")

    logger.info("event=rag_file_delete_started file_id=%s namespace=%s", file_id, table_name)

    conn = connect()
    cursor = conn.cursor()

    try:
        chunks_deleted = 0
        validate_query, validate_params = build_get_rag_file_for_delete_query(
            file_id, table_name
        )
        cursor.execute(validate_query, validate_params)
        if cursor.fetchone() is None:
            conn.rollback()
            return {"row_deleted": False, "chunks_deleted": 0}

        current_schema = schema
        table_exists_for_namespace = False
        if current_schema:
            exists_query, exists_params = build_check_table_exists_query(
                current_schema, table_name
            )
            cursor.execute(exists_query, exists_params)
            table_exists_for_namespace = cursor.fetchone() is not None

        if table_exists_for_namespace:
            chunk_query, chunk_params = build_delete_pgvector_by_file_id_query(
                table_name, file_id
            )
            cursor.execute(chunk_query, chunk_params)
            chunks_deleted = cursor.rowcount

        row_query, row_params = build_delete_rag_file_query(file_id, table_name)
        cursor.execute(row_query, row_params)
        row_deleted = cursor.rowcount > 0
        if not row_deleted:
            conn.rollback()
            return {"row_deleted": False, "chunks_deleted": 0}

        conn.commit()
        return {"row_deleted": row_deleted, "chunks_deleted": chunks_deleted}
    except Exception:
        conn.rollback()
        logger.exception("event=rag_file_delete_failed")
        raise
    finally:
        conn.close()


def table_exists(table_schema: str, table_name: str) -> bool:
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_check_table_exists_query(table_schema, table_name)
        cursor.execute(query, params)
        return cursor.fetchone() is not None
    except Exception as e:
        logger.exception("event=table_exists_check_failed")
        return False
    finally:
        conn.close()


def column_exists(table_schema: str, table_name: str, column_name: str) -> bool:
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_check_column_exists_query(table_schema, table_name, column_name)
        cursor.execute(query, params)
        return cursor.fetchone() is not None
    except Exception:
        logger.exception("event=column_exists_check_failed")
        return False
    finally:
        conn.close()


class SchemaChangeResult(Enum):
    """Outcome of a safe additive schema change, distinguishing a no-op
    from an applied change from a failure the caller must not treat as
    success."""

    CHANGED = "changed"
    UNCHANGED = "unchanged"
    FAILED = "failed"


# Fixed allow-list of column type/definition fragments. add_column_if_missing()
# only accepts a key from this map, never a raw type string, so it cannot be
# used as an unrestricted DDL injection surface.
_ALLOWED_COLUMN_TYPES: Dict[str, sql.SQL] = {
    "TEXT": sql.SQL("TEXT"),
    "INTEGER": sql.SQL("INTEGER"),
}


def _column_exists_strict(cursor, table_schema: str, table_name: str, column_name: str) -> bool:
    """
    Strict column-existence check for use inside schema-mutation control flow.

    Unlike column_exists(), this does not catch exceptions: a failed
    inspection must propagate to the caller instead of being collapsed into
    "column absent", because that ambiguity is exactly what would let a
    failed inspection incorrectly authorize an ALTER TABLE.
    """
    query, params = build_check_column_exists_query(table_schema, table_name, column_name)
    cursor.execute(query, params)
    return cursor.fetchone() is not None


def add_column_if_missing(
    table_schema: str, table_name: str, column_name: str, column_type: str
) -> SchemaChangeResult:
    """
    Adds a column to an existing table, only when its absence has been
    positively established. Fails closed: if the pre-DDL inspection cannot
    establish that the column is absent, no ALTER TABLE is executed.

    :param table_schema: Schema name.
    :param table_name: Table name.
    :param column_name: Name of the column to add.
    :param column_type: Key into the fixed allow-list of supported column
        type/definition fragments (see _ALLOWED_COLUMN_TYPES).
    :return: SchemaChangeResult.UNCHANGED if the column already existed,
        CHANGED if it was added and verified, FAILED if inspection, DDL, or
        post-DDL verification did not succeed.
    """
    if column_type not in _ALLOWED_COLUMN_TYPES:
        raise ValueError(f"Unsupported column type: {column_type}")

    conn = connect()
    cursor = conn.cursor()

    try:
        try:
            already_exists = _column_exists_strict(cursor, table_schema, table_name, column_name)
        except Exception:
            logger.exception(
                "event=add_column_if_missing_inspection_failed table=%s column=%s",
                table_name,
                column_name,
            )
            return SchemaChangeResult.FAILED

        if already_exists:
            return SchemaChangeResult.UNCHANGED

        try:
            query, params = build_add_column_query(
                table_schema, table_name, column_name, _ALLOWED_COLUMN_TYPES[column_type]
            )
            cursor.execute(query, params)
        except Exception:
            conn.rollback()
            logger.exception(
                "event=add_column_if_missing_ddl_failed table=%s column=%s",
                table_name,
                column_name,
            )
            return SchemaChangeResult.FAILED

        try:
            added = _column_exists_strict(cursor, table_schema, table_name, column_name)
        except Exception:
            conn.rollback()
            logger.exception(
                "event=add_column_if_missing_verification_failed table=%s column=%s",
                table_name,
                column_name,
            )
            return SchemaChangeResult.FAILED

        if not added:
            conn.rollback()
            logger.error(
                "event=add_column_if_missing_verification_mismatch table=%s column=%s",
                table_name,
                column_name,
            )
            return SchemaChangeResult.FAILED

        conn.commit()
        return SchemaChangeResult.CHANGED
    finally:
        cursor.close()
        conn.close()


def add_usage_service_column() -> None:
    """
    Governed, application-owned schema operation for the first Usage Service
    schema evolution: adds the nullable 'service' column to the existing
    'logs' table if it is not already present.

    Fixed intent (schema, table, column, type are not caller-controlled):
    current Maui schema / logs / service / TEXT. This is the only sanctioned
    way to reach add_column_if_missing() for this change; it does not accept
    schema/table/column/type parameters, so it cannot be used to mutate an
    arbitrary table or column.

    :raises RuntimeError: if the schema change was not committed (FAILED),
        so a failure is visible as a process failure rather than a silent
        success.
    """
    result = add_column_if_missing(schema, "logs", "service", "TEXT")

    if result == SchemaChangeResult.CHANGED:
        print("logs.service added.")
    elif result == SchemaChangeResult.UNCHANGED:
        print("logs.service already present, no change needed.")
    else:
        raise RuntimeError("Failed to add logs.service column.")


def add_usage_request_id_column() -> None:
    """
    Governed, application-owned schema operation for the Usage request_id
    schema evolution: adds the nullable 'request_id' column to the existing
    'logs' table if it is not already present.

    Fixed intent (schema, table, column, type are not caller-controlled):
    current Maui schema / logs / request_id / TEXT. This is the only
    sanctioned way to reach add_column_if_missing() for this change; it does
    not accept schema/table/column/type parameters, so it cannot be used to
    mutate an arbitrary table or column.

    :raises RuntimeError: if the schema change was not committed (FAILED),
        so a failure is visible as a process failure rather than a silent
        success.
    """
    result = add_column_if_missing(schema, "logs", "request_id", "TEXT")

    if result == SchemaChangeResult.CHANGED:
        print("logs.request_id added.")
    elif result == SchemaChangeResult.UNCHANGED:
        print("logs.request_id already present, no change needed.")
    else:
        raise RuntimeError("Failed to add logs.request_id column.")


def pgvector_maui_id_exists(table_name: str, maui_id: str) -> bool:
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_check_pgvector_maui_id_exists_query(table_name, maui_id)
        cursor.execute(query, params)
        return cursor.fetchone() is not None
    except Exception as e:
        logger.exception("event=pgvector_maui_id_check_failed")
        return False
    finally:
        conn.close()


def get_prompt_from_db(title: str, version: Optional[int] = None) -> Optional[str]:
    """
    Retrieves a prompt's message from the database by title, optionally filtering by version.

    :param title: The identifier of the prompt (e.g., 'start_chat_system').
    :param version: Optional specific version to retrieve. If None, retrieves the most recent version.
    :return: The prompt message as a string, or None if not found.
    """
    logger.info("event=prompt_lookup_started title=%s version=%s", title, version)

    conn = connect()
    cursor = conn.cursor()

    query, params = build_get_prompt_query(title, version)

    try:
        cursor.execute(query, params)
        result = cursor.fetchone()
        if result:
            return result[0]  # message
        else:
            logger.warning(
                "event=prompt_lookup_not_found title=%s version=%s",
                title,
                version,
            )
            return None
    except Exception as e:
        logger.exception("event=prompt_lookup_failed error=%s", str(e))
        return None
    finally:
        conn.close()


def save_feedback(
    user_email: str,
    question: str,
    answer: str,
    feedback_value: str,
    log_id: Optional[int] = None,
    source: Optional[str] = None,
) -> int:
    """
    Persists a feedback entry into the database.

    :param user_email: Username (email) of the user submitting the feedback.
    :param question: The question being evaluated.
    :param answer: The answer being evaluated.
    :param feedback_value: Feedback value ('positive' or 'negative').
    :param log_id: Optional reference to logs.id.
    :param source: Optional source identifier.
    :return: The generated feedback ID.
    """
    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_insert_feedback_query(
            user_email=user_email,
            question=question,
            answer=answer,
            feedback_value=feedback_value,
            log_id=log_id,
            source=source,
        )

        cursor.execute(query, params)

        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("Failed to retrieve feedback_id after insert")

        feedback_id = row[0]

        conn.commit()
        return feedback_id

    except Exception:
        conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()


def get_users_for_admin(page=1, limit=50, search=None):
    """
    Retrieves users from the database and returns them as a list of dictionaries
    for use in the admin panel with pagination.

    :param page: Page number (1-based)
    :param limit: Maximum number of users per page
    :return: Dictionary with user list and pagination info
    """
    conn = connect()
    cursor = conn.cursor()

    offset = (page - 1) * limit

    try:
        # Get users
        query, params = build_list_users_query(limit, offset, search=search)
        cursor.execute(query, params)
        users_raw = cursor.fetchall()

        # Get total count
        query_count, params_count = build_get_total_users_count_query(search=search)
        cursor.execute(query_count, params_count)
        total_count = cursor.fetchone()[0]

    finally:
        conn.close()

    users = []
    if users_raw:
        for id, username, api_key, date_valid_until, tokens in users_raw:
            try:
                decrypted_api_key = get_cipher_suite().decrypt(api_key).decode()
            except InvalidToken:
                decrypted_api_key = "Decryption failed"

            is_active = False
            # Formatta la data se esiste
            if date_valid_until and hasattr(date_valid_until, "strftime"):
                formatted_date = date_valid_until.strftime("%Y-%m-%d")
            else:
                formatted_date = str(date_valid_until) if date_valid_until else "N/A"

            # Check if user is active (today < date_valid_until)
            if date_valid_until:
                try:
                    # Se è una stringa, prova a parsarla
                    if isinstance(date_valid_until, str):
                        # Prova vari formati
                        for fmt in [
                            "%Y-%m-%d %H:%M:%S.%f",
                            "%Y-%m-%d %H:%M:%S",
                            "%Y-%m-%d",
                        ]:
                            try:
                                valid_date = datetime.strptime(
                                    date_valid_until, fmt
                                ).date()
                                break
                            except ValueError:
                                continue
                        else:
                            # Se nessun formato funziona
                            valid_date = None

                        if valid_date:
                            formatted_date = valid_date.strftime("%Y-%m-%d")
                    # Se è già un datetime
                    elif hasattr(date_valid_until, "date"):
                        valid_date = date_valid_until.date()
                        formatted_date = valid_date.strftime("%Y-%m-%d")
                    # Se è già un date
                    elif hasattr(date_valid_until, "year"):
                        valid_date = date_valid_until
                        formatted_date = valid_date.strftime("%Y-%m-%d")
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

            users.append(
                {
                    "id": id,
                    "name": username,
                    "email": decrypted_api_key,
                    "tokens": f"{tokens} tokens",
                    "is_active": is_active,
                    "created_at": formatted_date,
                }
            )

    total_pages = (total_count + limit - 1) // limit

    return {
        "users": users,
        "page": page,
        "total_pages": total_pages,
        "total_count": total_count,
    }


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

    return {"total_users": total_users, "total_tokens": total_tokens}


def get_logs_for_admin(page=1, limit=50, start_date=None, end_date=None, search=None):
    """
    Retrieves logs from the database for the admin panel with pagination.

    :param page: Page number (1-based)
    :param limit: Maximum number of logs per page
    :param start_date: Optional start date filter
    :param end_date: Optional end date filter
    :return: Dictionary with logs list and pagination info
    """
    conn = connect()
    cursor = conn.cursor()

    offset = (page - 1) * limit

    try:
        # Get logs
        query, params = build_get_logs_for_admin_query(
            limit, offset, start_date, end_date, search=search
        )
        cursor.execute(query, params)
        logs_raw = cursor.fetchall()

        # Get total count
        query_count, params_count = build_get_total_logs_count_query(
            start_date, end_date, search=search
        )
        cursor.execute(query_count, params_count)
        total_count = cursor.fetchone()[0]

    finally:
        conn.close()

    logs = []
    if logs_raw:
        for (
            log_id,
            user_id,
            username,
            date,
            token_input,
            token_output,
            cost,
            model,
            provider,
            service,
        ) in logs_raw:
            # Formatta la data
            if date and hasattr(date, "strftime"):
                formatted_date = date.strftime("%Y-%m-%d %H:%M:%S")
            else:
                formatted_date = str(date) if date else "N/A"

            logs.append(
                {
                    "id": log_id,
                    "user_id": user_id,
                    "username": username or "Unknown",
                    "date": formatted_date,
                    "token_input": token_input or 0,
                    "token_output": token_output or 0,
                    "cost": cost or 0,
                    "model": model or "N/A",
                    "provider": provider or "N/A",
                    "service": service or "N/A",
                }
            )

    total_pages = (total_count + limit - 1) // limit

    return {
        "logs": logs,
        "page": page,
        "total_pages": total_pages,
        "total_count": total_count,
    }


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
        one_year_from_today = datetime.now().replace(microsecond=0) + timedelta(
            days=365
        )

        query, params = build_update_user_tokens_query(
            user_id, new_tokens, one_year_from_today
        )
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


def get_logs_stats(start_date=None, end_date=None):
    """
    Get statistics about logs for charts and dashboard.

    :param start_date: Optional start date for stats
    :param end_date: Optional end date for stats
    :return: Dictionary with log statistics
    """
    conn = connect()
    cursor = conn.cursor()

    try:
        # Total tokens input/output
        query, params = build_get_total_log_stats_query(start_date, end_date)
        cursor.execute(query, params)
        totals = cursor.fetchone()

        # Tokens by day (filtered by date range)
        query, params = build_get_daily_log_stats_query(start_date, end_date)
        cursor.execute(query, params)
        daily_stats = cursor.fetchall()

        # Top users by token usage
        query, params = build_get_top_users_by_token_usage_query(start_date, end_date)
        cursor.execute(query, params)
        top_users = cursor.fetchall()

    finally:
        conn.close()

    # Format daily stats
    daily_data = []
    if daily_stats:
        for day, input_t, output_t in daily_stats:
            day_str = day.strftime("%Y-%m-%d") if hasattr(day, "strftime") else str(day)
            daily_data.append(
                {"day": day_str, "input": input_t or 0, "output": output_t or 0}
            )

    # Format top users
    top_users_data = []
    if top_users:
        for username, total in top_users:
            top_users_data.append(
                {"username": username or "Unknown", "total_tokens": total or 0}
            )

    return {
        "total_input": totals[0] if totals and totals[0] is not None else 0,
        "total_output": totals[1] if totals and totals[1] is not None else 0,
        "total_cost": totals[2] if totals and totals[2] is not None else 0.0,
        "total_requests": totals[3] if totals and totals[3] is not None else 0,
        "daily_stats": daily_data,
        "top_users": top_users_data,
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
                decrypted_api_key = get_cipher_suite().decrypt(api_key).decode()
            except InvalidToken:
                decrypted_api_key = "Decryption failed"

            # Formatta la data se esiste
            if date_valid_until and hasattr(date_valid_until, "strftime"):
                formatted_date = date_valid_until.strftime("%Y-%m-%d")
            else:
                formatted_date = str(date_valid_until) if date_valid_until else "N/A"

            return {
                "id": id,
                "username": username,
                "api_key": decrypted_api_key,
                "date_valid_until": formatted_date,
                "tokens": tokens,
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
            prompts.append(
                {
                    "id": id,
                    "title": title,
                    "version": version,
                    "message": message,
                }
            )

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
                "id": id,
                "title": title,
                "version": version,
                "message": message,
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
        logger.exception("event=cost_add_failed")
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
        logger.exception("event=cost_update_failed")
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
        logger.exception("event=cost_delete_failed")
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
            costs.append(
                {
                    "id": id,
                    "model": model,
                    "provider": provider,
                    "token_input_cost": in_cost,
                    "token_output_cost": out_cost,
                    "start_date_valid": start,
                    "end_date_valid": end,
                }
            )

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
                "id": id,
                "model": model,
                "provider": provider,
                "token_input_cost": in_cost,
                "token_output_cost": out_cost,
                "start_date_valid": start,
                "end_date_valid": end,
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

        return {"total_tokens": total_tokens, "total_cost": total_cost}
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
    logger.info("event=prompt_add_started title=%s", title)

    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_add_prompt_query(title, version, message)
        cursor.execute(query, params)
        conn.commit()
        return None
    except psycopg.IntegrityError as e:
        logger.warning("event=prompt_add_conflict title=%s error=%s", title, str(e))
        return f"Error adding new prompt: {e}"
    except Exception as e:
        logger.exception("event=prompt_add_failed")
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
    logger.info("event=prompt_delete_started prompt_id=%s", prompt_id)

    conn = connect()
    cursor = conn.cursor()

    try:
        query, params = build_delete_prompt_query(prompt_id)
        cursor.execute(query, params)
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.exception("event=prompt_delete_failed error=%s", e)
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
                activities.append({"type": type, "date": date, "details": details})
        return activities
    finally:
        conn.close()


def get_feedback_for_admin(
    source_filter: Optional[str] = None,
    page=1,
    limit=50,
    start_date=None,
    end_date=None,
) -> Dict[str, Any]:
    """
    Retrieves feedback entries for the admin panel with pagination.

    :param source_filter: Optional source to filter by.
    :param page: Page number (1-based)
    :param limit: Maximum number of entries per page
    :param start_date: Optional start date filter
    :param end_date: Optional end date filter
    :return: Dictionary with feedback list and pagination info.
    """
    conn = connect()
    cursor = conn.cursor()

    offset = (page - 1) * limit

    try:
        # Get feedback
        query, params = build_get_feedback_for_admin_query(
            limit, offset, source_filter, start_date, end_date
        )
        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Get total count
        query_count, params_count = build_get_total_feedback_count_query(
            source_filter, start_date, end_date
        )
        cursor.execute(query_count, params_count)
        total_count = cursor.fetchone()[0]

        feedback_list = []
        if rows:
            for row in rows:
                feedback_list.append(
                    {
                        "id": row[0],
                        "user_email": row[1],
                        "question": row[2],
                        "answer": row[3],
                        "feedback_value": row[4],
                        "timestamp": row[5],
                        "log_id": row[6],
                        "source": row[7],
                        "model": row[8],
                    }
                )

        total_pages = (total_count + limit - 1) // limit

        return {
            "feedbacks": feedback_list,
            "page": page,
            "total_pages": total_pages,
            "total_count": total_count,
        }
    except Exception as e:
        logger.exception("event=feedback_admin_lookup_failed error=%s", e)
        return {"feedbacks": [], "page": 1, "total_pages": 1, "total_count": 0}
    finally:
        conn.close()


def get_feedback_stats(
    source_filter: Optional[str] = None, start_date=None, end_date=None
) -> Dict[str, Any]:
    """
    Retrieves feedback statistics and list of sources.

    :param source_filter: Optional source to filter by.
    :param start_date: Optional start date filter
    :param end_date: Optional end date filter
    :return: Dictionary containing stats and sources.
    """
    conn = connect()
    cursor = conn.cursor()

    stats = {
        "total": 0,
        "positive_count": 0,
        "negative_count": 0,
        "sources": [],
        "models": [],
    }

    try:
        # Get counts
        query_stats, params_stats = build_get_feedback_stats_query(
            source_filter, start_date, end_date
        )
        cursor.execute(query_stats, params_stats)
        row = cursor.fetchone()

        if row:
            stats["total"] = row[0]
            stats["positive_count"] = row[1]
            stats["negative_count"] = row[2]

        # Get sources (always get all sources for the filter dropdown)
        query_sources, _ = build_get_feedback_sources_query()
        cursor.execute(query_sources)
        rows_sources = cursor.fetchall()
        if rows_sources:
            stats["sources"] = [r[0] for r in rows_sources]

        # Get model stats
        query_models, params_models = build_get_feedback_model_stats_query(
            source_filter, start_date, end_date
        )
        cursor.execute(query_models, params_models)
        rows_models = cursor.fetchall()
        if rows_models:
            stats["models"] = [
                {"name": r[0] or "Unknown", "count": r[1]} for r in rows_models
            ]

        return stats
    except Exception as e:
        logger.exception("event=feedback_stats_lookup_failed error=%s", e)
        return stats
    finally:
        conn.close()


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
    print("  add_usage_service_column    Add the nullable logs.service column if missing")
    print("  add_usage_request_id_column Add the nullable logs.request_id column if missing")


def _resolve_cli_command(argv: list[str]):
    """
    Validate argv against the known CLI commands and their expected argument
    counts, without executing anything.

    :return: A zero-argument callable that runs the selected command when
        invoked, or None if argv does not match any known command with the
        right argument count (help should be shown, no DB init required).
    """
    if len(argv) <= 1:
        return None

    command = argv[1]

    if command == "init_db":
        return init_db
    if command == "add_user" and len(argv) == 4:
        username, api_key = argv[2], argv[3]
        return lambda: add_user(username, api_key)
    if command == "remove_user" and len(argv) == 3:
        username = argv[2]
        return lambda: remove_user(username)
    if command == "get_user_by_username" and len(argv) == 3:
        user_name = argv[2]
        return lambda: get_user_by_username(user_name)
    if command == "edit_tokens" and len(argv) == 4:
        username, tokens_quantity = argv[2], int(argv[3])
        return lambda: edit_tokens(username, tokens_quantity)
    if command == "list_users":
        return list_users
    if command == "print_keys":
        return print_stored_keys
    if command == "add_usage_service_column" and len(argv) == 2:
        return add_usage_service_column
    if command == "add_usage_request_id_column" and len(argv) == 2:
        return add_usage_request_id_column

    return None


def run_cli(argv: list[str]) -> None:
    """
    CLI entry point for direct script invocation.

    Validates the requested command and its argument count first; only a
    syntactically valid, known DB command triggers .env loading,
    configuration loading and database_pg initialization. Help and invalid
    invocations never touch load_dotenv()/load_config()/init(), so they
    don't require DB credentials or ENCRYPTION_KEY to be set.
    """
    command = _resolve_cli_command(argv)
    if command is None:
        print_help()
        return

    load_dotenv()
    init(load_config())
    command()


if __name__ == "__main__":
    run_cli(sys.argv)
