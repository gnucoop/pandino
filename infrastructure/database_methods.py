"""
This module contains functions for safely building SQL queries.
Each function returns a composable SQL object (psycopg.sql) and its parameters.
"""

from psycopg import sql
from typing import Tuple, Any, Optional, List


def build_get_user_by_username_query(username: str) -> Tuple[sql.Composed, Tuple[str]]:
    """
    Builds a SQL query to select a user from the 'users' table by username.

    :param username: The username to search for in the 'users' table.
    :return: A tuple containing the SQL query object and a tuple of parameters for query execution.
    """
    query = sql.SQL("SELECT * FROM {table} WHERE {col_username} = %s").format(
        table=sql.Identifier("users"), col_username=sql.Identifier("username")
    )
    return query, (username,)


def build_add_user_query(
    username: str,
    encrypted_api_key: str,
    date_valid_until: str,
    client: Optional[str] = None,
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to insert a new user into the 'users' table.

    :param username: Unique username of the user.
    :param encrypted_api_key: API key already encrypted as string.
    :param date_valid_until: Expiration date in ISO format.
    :param client: Authenticated client to persist at creation time, or
        None when not known (e.g. CLI-created users).
    :return: Tuple with SQL query and parameters.
    """
    query = sql.SQL(
        "INSERT INTO {table} ({col_username}, {col_api_key}, {col_date}, {col_client}) "
        "VALUES (%s, %s, %s, %s)"
    ).format(
        table=sql.Identifier("users"),
        col_username=sql.Identifier("username"),
        col_api_key=sql.Identifier("api_key"),
        col_date=sql.Identifier("date_valid_until"),
        col_client=sql.Identifier("client"),
    )
    params = (username, encrypted_api_key, date_valid_until, client)
    return query, params


def build_remove_user_query(username: str) -> Tuple[sql.Composed, Tuple[str]]:
    """
    Builds a SQL query to delete a user from the 'users' table by username.

    :param username: The username of the user to remove.
    :return: A tuple with the SQL query and parameters.
    """
    query = sql.SQL("DELETE FROM {table} WHERE {col_username} = %s").format(
        table=sql.Identifier("users"), col_username=sql.Identifier("username")
    )
    return query, (username,)


def build_edit_tokens_query(
    tokens_quantity: int, date_valid_until: str, username: str
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to update a user's token balance and expiration date.

    :param tokens_quantity: Number of tokens to add (can be negative).
    :param date_valid_until: New expiration date (ISO format).
    :param username: The username of the user to update.
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        "UPDATE {table} "
        "SET {col_tokens} = {col_tokens} + %s, {col_date} = %s "
        "WHERE {col_username} = %s"
    ).format(
        table=sql.Identifier("users"),
        col_tokens=sql.Identifier("tokens"),
        col_date=sql.Identifier("date_valid_until"),
        col_username=sql.Identifier("username"),
    )
    params = (tokens_quantity, date_valid_until, username)
    return query, params


def build_list_users_query(
    limit: int, offset: int = 0, search: Optional[str] = None
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to retrieve users with selected columns from the 'users' table with pagination.

    :param limit: Maximum number of users to retrieve.
    :param offset: Number of users to skip.
    :param search: Optional search term to filter by username.
    :return: Tuple with SQL query and parameters.
    """
    where_clause = sql.SQL("")
    params: List[Any] = []

    if search:
        where_clause = sql.SQL("WHERE {username} ILIKE %s").format(username=sql.Identifier("username"))
        params.append(f"%{search}%")

    params.extend([limit, offset])

    query = sql.SQL(
        "SELECT {id}, {username}, {api_key}, {date_valid_until}, {tokens} FROM {table} {where_clause} ORDER BY {id} ASC LIMIT %s OFFSET %s"
    ).format(
        id=sql.Identifier("id"),
        username=sql.Identifier("username"),
        api_key=sql.Identifier("api_key"),
        date_valid_until=sql.Identifier("date_valid_until"),
        tokens=sql.Identifier("tokens"),
        table=sql.Identifier("users"),
        where_clause=where_clause,
    )
    return query, tuple(params)


def build_get_total_users_count_query(search: Optional[str] = None) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to count total users for pagination.

    :param search: Optional search term to filter by username.
    :return: Tuple with SQL query and parameter tuple.
    """
    where_clause = sql.SQL("")
    params: Tuple[Any, ...] = ()

    if search:
        where_clause = sql.SQL("WHERE {username} ILIKE %s").format(username=sql.Identifier("username"))
        params = (f"%{search}%",)

    query = sql.SQL("SELECT COUNT(*) FROM {table} {where_clause}").format(
        table=sql.Identifier("users"),
        where_clause=where_clause,
    )
    return query, params


def build_check_table_exists_query(
    table_schema: str,
    table_name: str,
) -> Tuple[sql.SQL, Tuple[Any, ...]]:
    """
    Builds a SQL query to check whether a table exists in a given schema.

    :param table_schema: Schema name.
    :param table_name: Table name.
    :return: Tuple with SQL query and parameters.
    """
    query = sql.SQL(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = %s
          AND table_name = %s
        LIMIT 1
        """
    )
    params = (table_schema, table_name)
    return query, params


def build_check_column_exists_query(
    table_schema: str,
    table_name: str,
    column_name: str,
) -> Tuple[sql.SQL, Tuple[Any, ...]]:
    """
    Builds a SQL query to check whether a column exists on a given table.

    :param table_schema: Schema name.
    :param table_name: Table name.
    :param column_name: Column name.
    :return: Tuple with SQL query and parameters.
    """
    query = sql.SQL(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
          AND column_name = %s
        LIMIT 1
        """
    )
    params = (table_schema, table_name, column_name)
    return query, params


def build_add_column_query(
    table_schema: str,
    table_name: str,
    column_name: str,
    column_type_sql: sql.SQL,
) -> Tuple[sql.Composed, Tuple[()]]:
    """
    Builds a SQL query to add a new column to an existing table.

    :param table_schema: Schema name.
    :param table_name: Table name.
    :param column_name: Name of the column to add.
    :param column_type_sql: Pre-validated sql.SQL fragment for the column
        type/definition (e.g. sql.SQL("TEXT")). Callers must resolve this
        from a fixed allow-list rather than passing a raw string, so this
        function never builds an unrestricted DDL injection surface.
    :return: Tuple with SQL query and empty parameter tuple.
    """
    query = sql.SQL("ALTER TABLE {table} ADD COLUMN {column} {type}").format(
        table=sql.Identifier(table_schema, table_name),
        column=sql.Identifier(column_name),
        type=column_type_sql,
    )
    return query, ()


def build_check_pgvector_maui_id_exists_query(
    table_name: str,
    maui_id: str,
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to check whether a row with the given maui_id
    already exists inside a PGVector namespace table.

    :param table_name: Dynamic PGVector table name.
    :param maui_id: Deterministic Maui paragraph ID.
    :return: Tuple with SQL query and parameters.
    """
    query = sql.SQL(
        """
        SELECT 1
        FROM {table}
        WHERE langchain_metadata->>'maui_id' = %s
        LIMIT 1
        """
    ).format(
        table=sql.Identifier(table_name),
    )
    params = (maui_id,)
    return query, params


def build_insert_rag_file_query(
    file_id: str,
    file_name: str,
    namespace: str,
    chunk_count: int,
    language: Optional[str],
) -> Tuple[sql.SQL, Tuple[Any, ...]]:
    """
    Builds a SQL query to insert a new record into the rag_files table.

    :param file_id: Unique identifier of the document (file-level, hash-based).
    :param file_name: Original name of the uploaded file.
    :param namespace: Namespace (table) where embeddings are stored.
    :param chunk_count: Number of chunks generated from the document.
    :param language: Language of the document.
    :return: Tuple with SQL query and parameters.
    """
    query = sql.SQL(
        """
        INSERT INTO rag_files (id, file_name, namespace, chunk_count, language)
        VALUES (%s, %s, %s, %s, %s)
        """
    )
    params = (file_id, file_name, namespace, chunk_count, language)
    return query, params


def build_get_all_rag_files_query() -> Tuple[sql.Composed, Tuple[()]]:
    """
    Builds a SQL query to retrieve all records from the rag_files table.

    :return: Tuple with SQL query and empty parameter tuple.
    """
    query = sql.SQL(
        "SELECT id, file_name, namespace, chunk_count, language, created_at "
        "FROM rag_files ORDER BY created_at DESC"
    )
    return query, ()


def build_get_rag_file_for_delete_query(
    file_id: str,
    namespace: str,
) -> Tuple[sql.SQL, Tuple[Any, ...]]:
    """
    Builds a SQL query to lock and validate a RAG file before deletion.

    :param file_id: Unique identifier of the document (file-level, hash-based).
    :param namespace: Normalized namespace where embeddings are stored.
    :return: Tuple with SQL query and parameters.
    """
    query = sql.SQL(
        "SELECT id FROM rag_files WHERE id = %s AND namespace = %s FOR UPDATE"
    )
    params = (file_id, namespace)
    return query, params


def build_delete_rag_file_query(
    file_id: str,
    namespace: str,
) -> Tuple[sql.SQL, Tuple[Any, ...]]:
    """
    Builds a SQL query to delete a record from the rag_files table by its id
    and normalized namespace.

    :param file_id: Unique identifier of the document (file-level, hash-based).
    :param namespace: Normalized namespace where embeddings are stored.
    :return: Tuple with SQL query and parameters.
    """
    query = sql.SQL("DELETE FROM rag_files WHERE id = %s AND namespace = %s")
    params = (file_id, namespace)
    return query, params


def build_delete_pgvector_by_file_id_query(
    table_name: str,
    file_id: str,
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to delete all chunks of a document from a PGVector
    namespace table, matched by the file_id stored in chunk metadata.

    :param table_name: Dynamic PGVector table name (normalized namespace).
    :param file_id: File-level identifier stored in each chunk's metadata.
    :return: Tuple with SQL query and parameters.
    """
    query = sql.SQL(
        """
        DELETE FROM {table}
        WHERE langchain_metadata->>'file_id' = %s
        """
    ).format(
        table=sql.Identifier(table_name),
    )
    params = (file_id,)
    return query, params


def build_print_stored_keys_query() -> Tuple[sql.Composed, Tuple[()]]:
    """
    Builds a SQL query to retrieve usernames and encrypted API keys from the 'users' table.

    :return: Tuple with SQL query and empty parameter tuple.
    """
    query = sql.SQL("SELECT {col_user}, {col_key} FROM {table}").format(
        col_user=sql.Identifier("username"),
        col_key=sql.Identifier("api_key"),
        table=sql.Identifier("users"),
    )
    return query, ()


def build_validate_api_key_query(username: str) -> Tuple[sql.Composed, Tuple[str]]:
    """
    Builds a SQL query to retrieve API keys, expiration dates and client for a given user.

    :param username: The username to search for.
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        "SELECT {col_key}, {col_date}, {col_client} FROM {table} WHERE {col_user} = %s"
    ).format(
        col_key=sql.Identifier("api_key"),
        col_date=sql.Identifier("date_valid_until"),
        col_client=sql.Identifier("client"),
        table=sql.Identifier("users"),
        col_user=sql.Identifier("username"),
    )
    return query, (username,)


def build_get_token_cost_query(
    provider: str, model: str, current_date: str
) -> Tuple[sql.Composed, Tuple[str, str, str, str]]:
    """
    Builds a query to fetch token input/output cost for a specific provider and model, valid on current_date.

    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        "SELECT {in_cost}, {out_cost} FROM {table} "
        "WHERE {col_provider} = %s AND {col_model} = %s "
        "AND {col_start} <= %s AND {col_end} >= %s"
    ).format(
        in_cost=sql.Identifier("token_input_cost"),
        out_cost=sql.Identifier("token_output_cost"),
        table=sql.Identifier("costs"),
        col_provider=sql.Identifier("provider"),
        col_model=sql.Identifier("model"),
        col_start=sql.Identifier("start_date_valid"),
        col_end=sql.Identifier("end_date_valid"),
    )
    return query, (provider, model, current_date, current_date)


def build_insert_token_log_query(
    date: str,
    user_id: int,
    token_input: int,
    token_output: int,
    cost: float,
    model: str,
    provider: str,
    service: str,
    request_id: str,
    source: Optional[str],
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to insert a new usage log into the 'logs' table
    and returns the generated log ID.

    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        "INSERT INTO {table} ({col_date}, {col_user}, {col_in}, {col_out}, {col_cost}, {col_model}, {col_provider}, {col_service}, {col_request_id}, {col_source}) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "RETURNING id"
    ).format(
        table=sql.Identifier("logs"),
        col_date=sql.Identifier("date"),
        col_user=sql.Identifier("user_id"),
        col_in=sql.Identifier("token_input"),
        col_out=sql.Identifier("token_output"),
        col_cost=sql.Identifier("cost"),
        col_model=sql.Identifier("model"),
        col_provider=sql.Identifier("provider"),
        col_service=sql.Identifier("service"),
        col_request_id=sql.Identifier("request_id"),
        col_source=sql.Identifier("source"),
    )
    params = (
        date,
        user_id,
        token_input,
        token_output,
        cost,
        model,
        provider,
        service,
        request_id,
        source,
    )
    return query, params


def build_get_total_users_query() -> Tuple[sql.Composed, Tuple[()]]:
    """
    Builds a SQL query to count the total number of users.

    :return: Tuple with SQL query and empty parameter tuple.
    """
    query = sql.SQL("SELECT COUNT(*) FROM {table}").format(
        table=sql.Identifier("users")
    )
    return query, ()


def build_get_total_tokens_query() -> Tuple[sql.Composed, Tuple[()]]:
    """
    Builds a SQL query to sum the total number of tokens for all users.

    :return: Tuple with SQL query and empty parameter tuple.
    """
    query = sql.SQL("SELECT SUM({tokens}) FROM {table}").format(
        tokens=sql.Identifier("tokens"), table=sql.Identifier("users")
    )
    return query, ()


def build_get_logs_for_admin_query(
    limit: int,
    offset: int = 0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search: Optional[str] = None,
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to retrieve logs for the admin panel with pagination and date filtering.

    :param limit: Maximum number of logs to retrieve.
    :param offset: Number of logs to skip.
    :param start_date: Start date string (YYYY-MM-DD).
    :param end_date: End date string (YYYY-MM-DD).
    :param search: Optional search term to filter by username, model, or provider.
    :return: Tuple with SQL query and parameters.
    """
    conditions: List[sql.Composable] = []
    params: List[Any] = []

    if start_date and end_date:
        conditions.append(sql.SQL("l.date::timestamp >= %s::timestamp AND l.date::timestamp <= %s::timestamp"))
        params.extend([start_date, end_date + " 23:59:59"])

    if search:
        conditions.append(sql.SQL("(u.username ILIKE %s OR l.model ILIKE %s OR l.provider ILIKE %s)"))
        params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])

    where_clause = sql.SQL("")
    if conditions:
        where_clause = sql.SQL("WHERE ") + sql.SQL(" AND ").join(conditions)

    params.extend([limit, offset])

    query = sql.SQL(
        """
        SELECT l.id, l.user_id, u.username, l.date, l.token_input,
               l.token_output, l.cost, l.model, l.provider, l.service,
               l.request_id, l.duration_ms
        FROM {logs} l
        LEFT JOIN {users} u ON l.user_id = u.id
        {where_clause}
        ORDER BY l.date DESC
        LIMIT %s OFFSET %s
    """
    ).format(
        logs=sql.Identifier("logs"),
        users=sql.Identifier("users"),
        where_clause=where_clause,
    )
    return query, tuple(params)


def build_get_total_logs_count_query(
    start_date: Optional[str] = None, end_date: Optional[str] = None, search: Optional[str] = None
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to count total logs for pagination with date filtering.

    :param start_date: Start date string (YYYY-MM-DD).
    :param end_date: End date string (YYYY-MM-DD).
    :param search: Optional search term to filter by username, model, or provider.
    :return: Tuple with SQL query and parameter tuple.
    """
    conditions: List[sql.Composable] = []
    params: List[Any] = []

    if start_date and end_date:
        conditions.append(sql.SQL("l.date::timestamp >= %s::timestamp AND l.date::timestamp <= %s::timestamp"))
        params.extend([start_date, end_date + " 23:59:59"])

    if search:
        conditions.append(sql.SQL("(u.username ILIKE %s OR l.model ILIKE %s OR l.provider ILIKE %s)"))
        params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])

    where_clause = sql.SQL("")
    if conditions:
        where_clause = sql.SQL("WHERE ") + sql.SQL(" AND ").join(conditions)

    if search:
        query = sql.SQL("SELECT COUNT(*) FROM {logs} l LEFT JOIN {users} u ON l.user_id = u.id {where_clause}").format(
            logs=sql.Identifier("logs"),
            users=sql.Identifier("users"),
            where_clause=where_clause,
        )
    else:
        query = sql.SQL("SELECT COUNT(*) FROM {logs} l {where_clause}").format(
            logs=sql.Identifier("logs"),
            where_clause=where_clause,
        )
    return query, tuple(params)


def build_update_user_tokens_query(
    user_id: int, new_tokens: int, date_valid_until: Any
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to update the tokens and expiration date for a user.

    :param user_id: ID of the user to update.
    :param new_tokens: New token value.
    :param date_valid_until: New expiration date.
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        "UPDATE {table} SET {tokens} = %s, {date_valid_until} = %s WHERE {id} = %s"
    ).format(
        table=sql.Identifier("users"),
        tokens=sql.Identifier("tokens"),
        date_valid_until=sql.Identifier("date_valid_until"),
        id=sql.Identifier("id"),
    )
    return query, (new_tokens, date_valid_until, user_id)


def build_update_usage_duration_query(
    log_id: int, duration_ms: int
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to update the duration_ms of a logs row.

    :param log_id: ID of the log row to update.
    :param duration_ms: New duration value, in milliseconds.
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL("UPDATE {table} SET {duration_ms} = %s WHERE {id} = %s").format(
        table=sql.Identifier("logs"),
        duration_ms=sql.Identifier("duration_ms"),
        id=sql.Identifier("id"),
    )
    return query, (duration_ms, log_id)


def build_set_user_client_if_missing_query(
    username: str, client: str
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to set a user's client only if it is not already set.

    The fill-if-empty invariant is enforced by the WHERE clause itself
    (client IS NULL), so the row is only updated when no client has been
    persisted yet - no application-side read/decide/write ownership check.

    :param username: The username of the user to update.
    :param client: Candidate client value to persist.
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        "UPDATE {table} SET {client} = %s WHERE {username} = %s AND {client} IS NULL"
    ).format(
        table=sql.Identifier("users"),
        client=sql.Identifier("client"),
        username=sql.Identifier("username"),
    )
    return query, (client, username)


def build_get_total_log_stats_query(
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to get total log statistics with optional date filtering.

    :param start_date: Start date string (YYYY-MM-DD).
    :param end_date: End date string (YYYY-MM-DD).
    :return: Tuple with SQL query and parameter tuple.
    """
    where_clause = sql.SQL("")
    params: Tuple[Any, ...] = ()

    if start_date and end_date:
        where_clause = sql.SQL(
            "WHERE date::timestamp >= %s::timestamp AND date::timestamp <= %s::timestamp"
        )
        params = (start_date, end_date + " 23:59:59")

    query = sql.SQL(
        """
        SELECT
            COALESCE(SUM({token_input}), 0) as total_input,
            COALESCE(SUM({token_output}), 0) as total_output,
            COALESCE(SUM({cost}), 0.0) as total_cost,
            COUNT(*) as total_requests
        FROM {logs}
        {where_clause}
    """
    ).format(
        token_input=sql.Identifier("token_input"),
        token_output=sql.Identifier("token_output"),
        cost=sql.Identifier("cost"),
        logs=sql.Identifier("logs"),
        where_clause=where_clause,
    )
    return query, params


def build_get_daily_log_stats_query(
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to get daily log statistics for a specific date range.
    If no dates provided, defaults to last 7 days.

    :param start_date: Start date string (YYYY-MM-DD).
    :param end_date: End date string (YYYY-MM-DD).
    :return: Tuple with SQL query and parameter tuple.
    """

    if start_date and end_date:
        # User defined range
        where_clause = sql.SQL(
            "WHERE date::timestamp >= %s::timestamp AND date::timestamp <= %s::timestamp"
        )
        params = (start_date, end_date + " 23:59:59")  # Include full end day
    else:
        # Default last 7 days
        where_clause = sql.SQL(
            "WHERE date::timestamp >= CURRENT_DATE - INTERVAL '7 days'"
        )
        params = ()

    query = sql.SQL(
        """
        SELECT
            DATE(date::timestamp) as day,
            SUM({token_input}) as input_tokens,
            SUM({token_output}) as output_tokens
        FROM {logs}
        {where_clause}
        GROUP BY DATE(date::timestamp)
        ORDER BY day
    """
    ).format(
        token_input=sql.Identifier("token_input"),
        token_output=sql.Identifier("token_output"),
        logs=sql.Identifier("logs"),
        where_clause=where_clause,
    )
    return query, params


def build_get_top_users_by_token_usage_query(
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to get top 5 users by token usage with optional date filtering.

    :param start_date: Start date string (YYYY-MM-DD).
    :param end_date: End date string (YYYY-MM-DD).
    :return: Tuple with SQL query and parameter tuple.
    """
    where_clause = sql.SQL("")
    params: Tuple[Any, ...] = ()

    if start_date and end_date:
        where_clause = sql.SQL(
            "WHERE l.date::timestamp >= %s::timestamp AND l.date::timestamp <= %s::timestamp"
        )
        params = (start_date, end_date + " 23:59:59")

    query = sql.SQL(
        """
        SELECT
            u.username,
            SUM(l.token_input + l.token_output) as total_tokens
        FROM {logs} l
        LEFT JOIN {users} u ON l.user_id = u.id
        {where_clause}
        GROUP BY u.username
        ORDER BY total_tokens DESC
        LIMIT 5
    """
    ).format(
        logs=sql.Identifier("logs"),
        users=sql.Identifier("users"),
        where_clause=where_clause,
    )
    return query, params


def build_get_user_by_id_query(user_id: int) -> Tuple[sql.Composed, Tuple[int]]:
    """
    Builds a SQL query to retrieve a user by their ID.

    :param user_id: The ID of the user to retrieve.
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        "SELECT {id}, {username}, {api_key}, {date_valid_until}, {tokens} FROM {table} WHERE {id} = %s"
    ).format(
        id=sql.Identifier("id"),
        username=sql.Identifier("username"),
        api_key=sql.Identifier("api_key"),
        date_valid_until=sql.Identifier("date_valid_until"),
        tokens=sql.Identifier("tokens"),
        table=sql.Identifier("users"),
    )
    return query, (user_id,)


def build_get_prompt_query(
    title: str, version: Optional[int] = None
) -> Tuple[sql.Composed, Tuple[str, int] | Tuple[str]]:
    """
    Builds a SQL query to retrieve a prompt by title, optionally filtering by version.
    If version is None, returns the prompt with the highest version for that title.

    :param title: The identifier of the prompt (e.g., 'start_chat_system').
    :param version: Optional specific version to retrieve.
    :return: Tuple of SQL query and parameters.
    """
    if version is not None:
        query = sql.SQL(
            "SELECT {col_message} FROM {table} "
            "WHERE {col_title} = %s AND {col_version} = %s"
        ).format(
            col_message=sql.Identifier("message"),
            table=sql.Identifier("prompts"),
            col_title=sql.Identifier("title"),
            col_version=sql.Identifier("version"),
        )
        return query, (title, version)
    else:
        query = sql.SQL(
            "SELECT {col_message} FROM {table} "
            "WHERE {col_title} = %s "
            "ORDER BY {col_version} DESC LIMIT 1"
        ).format(
            col_message=sql.Identifier("message"),
            table=sql.Identifier("prompts"),
            col_title=sql.Identifier("title"),
            col_version=sql.Identifier("version"),
        )
        return query, (title,)


def build_get_all_prompts_query() -> Tuple[sql.Composed, Tuple[()]]:
    """
    Builds a SQL query to retrieve all prompts.

    :return: Tuple with SQL query and empty parameter tuple.
    """
    query = sql.SQL(
        "SELECT {id}, {title}, {version}, {message} FROM {table} ORDER BY {id} ASC"
    ).format(
        id=sql.Identifier("id"),
        title=sql.Identifier("title"),
        version=sql.Identifier("version"),
        message=sql.Identifier("message"),
        table=sql.Identifier("prompts"),
    )
    return query, ()


def build_get_prompt_by_id_query(prompt_id: int) -> Tuple[sql.Composed, Tuple[int]]:
    """
    Builds a SQL query to retrieve a prompt by its ID.

    :param prompt_id: The ID of the prompt to retrieve.
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        "SELECT {id}, {title}, {version}, {message} FROM {table} WHERE {id} = %s"
    ).format(
        id=sql.Identifier("id"),
        title=sql.Identifier("title"),
        version=sql.Identifier("version"),
        message=sql.Identifier("message"),
        table=sql.Identifier("prompts"),
    )
    return query, (prompt_id,)


def build_add_prompt_query(
    title: str, version: int, message: str
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to insert a new prompt into the 'prompts' table.

    :param title: Title of the prompt.
    :param version: Version of the prompt.
    :param message: Message of the prompt.
    :return: Tuple with SQL query and parameters.
    """
    query = sql.SQL(
        "INSERT INTO {table} ({col_title}, {col_version}, {col_message}) "
        "VALUES (%s, %s, %s)"
    ).format(
        table=sql.Identifier("prompts"),
        col_title=sql.Identifier("title"),
        col_version=sql.Identifier("version"),
        col_message=sql.Identifier("message"),
    )
    params = (title, version, message)
    return query, params


def build_update_prompt_query(
    prompt_id: int, title: str, version: int, message: str
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to update a prompt.

    :param prompt_id: ID of the prompt to update.
    :param title: New title value.
    :param version: New version value.
    :param message: New message value.
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        "UPDATE {table} SET {title} = %s, {version} = %s, {message} = %s WHERE {id} = %s"
    ).format(
        table=sql.Identifier("prompts"),
        title=sql.Identifier("title"),
        version=sql.Identifier("version"),
        message=sql.Identifier("message"),
        id=sql.Identifier("id"),
    )
    return query, (title, version, message, prompt_id)


def build_delete_prompt_query(prompt_id: int) -> Tuple[sql.Composed, Tuple[int]]:
    """
    Builds a SQL query to delete a prompt from the 'prompts' table by prompt_id.

    :param prompt_id: The id of the prompt to remove.
    :return: A tuple with the SQL query and parameters.
    """
    query = sql.SQL("DELETE FROM {table} WHERE {col_id} = %s").format(
        table=sql.Identifier("prompts"), col_id=sql.Identifier("id")
    )
    return query, (prompt_id,)


def build_add_cost_query(
    model: str,
    provider: str,
    token_input_cost: float,
    token_output_cost: float,
    start_date_valid: str,
    end_date_valid: str,
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to insert a new cost entry into the 'costs' table.

    :param model: Model name.
    :param provider: Provider name.
    :param token_input_cost: Cost per input token.
    :param token_output_cost: Cost per output token.
    :param start_date_valid: Start date of validity.
    :param end_date_valid: End date of validity.
    :return: Tuple with SQL query and parameters.
    """
    query = sql.SQL(
        "INSERT INTO {table} ({col_model}, {col_provider}, {col_in_cost}, {col_out_cost}, {col_start}, {col_end}) "
        "VALUES (%s, %s, %s, %s, %s, %s)"
    ).format(
        table=sql.Identifier("costs"),
        col_model=sql.Identifier("model"),
        col_provider=sql.Identifier("provider"),
        col_in_cost=sql.Identifier("token_input_cost"),
        col_out_cost=sql.Identifier("token_output_cost"),
        col_start=sql.Identifier("start_date_valid"),
        col_end=sql.Identifier("end_date_valid"),
    )
    params = (
        model,
        provider,
        token_input_cost,
        token_output_cost,
        start_date_valid,
        end_date_valid,
    )
    return query, params


def build_update_cost_query(
    cost_id: int,
    model: str,
    provider: str,
    token_input_cost: float,
    token_output_cost: float,
    start_date_valid: str,
    end_date_valid: str,
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to update a cost entry.

    :param cost_id: ID of the cost entry to update.
    :param model: New model name.
    :param provider: New provider name.
    :param token_input_cost: New input token cost.
    :param token_output_cost: New output token cost.
    :param start_date_valid: New start date.
    :param end_date_valid: New end date.
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        "UPDATE {table} SET {col_model} = %s, {col_provider} = %s, "
        "{col_in_cost} = %s, {col_out_cost} = %s, "
        "{col_start} = %s, {col_end} = %s "
        "WHERE {col_id} = %s"
    ).format(
        table=sql.Identifier("costs"),
        col_model=sql.Identifier("model"),
        col_provider=sql.Identifier("provider"),
        col_in_cost=sql.Identifier("token_input_cost"),
        col_out_cost=sql.Identifier("token_output_cost"),
        col_start=sql.Identifier("start_date_valid"),
        col_end=sql.Identifier("end_date_valid"),
        col_id=sql.Identifier("id"),
    )
    params = (
        model,
        provider,
        token_input_cost,
        token_output_cost,
        start_date_valid,
        end_date_valid,
        cost_id,
    )
    return query, params


def build_delete_cost_query(cost_id: int) -> Tuple[sql.Composed, Tuple[int]]:
    """
    Builds a SQL query to delete a cost entry from the 'costs' table by id.

    :param cost_id: The id of the cost entry to remove.
    :return: A tuple with the SQL query and parameters.
    """
    query = sql.SQL("DELETE FROM {table} WHERE {col_id} = %s").format(
        table=sql.Identifier("costs"), col_id=sql.Identifier("id")
    )
    return query, (cost_id,)


def build_get_all_costs_query() -> Tuple[sql.Composed, Tuple[()]]:
    """
    Builds a SQL query to retrieve all cost entries.

    :return: Tuple with SQL query and empty parameter tuple.
    """
    query = sql.SQL(
        "SELECT {id}, {model}, {provider}, {in_cost}, {out_cost}, {start}, {end} "
        "FROM {table} ORDER BY {id} ASC"
    ).format(
        id=sql.Identifier("id"),
        model=sql.Identifier("model"),
        provider=sql.Identifier("provider"),
        in_cost=sql.Identifier("token_input_cost"),
        out_cost=sql.Identifier("token_output_cost"),
        start=sql.Identifier("start_date_valid"),
        end=sql.Identifier("end_date_valid"),
        table=sql.Identifier("costs"),
    )
    return query, ()


def build_get_cost_by_id_query(cost_id: int) -> Tuple[sql.Composed, Tuple[int]]:
    """
    Builds a SQL query to retrieve a cost entry by its ID.

    :param cost_id: The ID of the cost entry to retrieve.
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        "SELECT {id}, {model}, {provider}, {in_cost}, {out_cost}, {start}, {end} "
        "FROM {table} WHERE {id} = %s"
    ).format(
        id=sql.Identifier("id"),
        model=sql.Identifier("model"),
        provider=sql.Identifier("provider"),
        in_cost=sql.Identifier("token_input_cost"),
        out_cost=sql.Identifier("token_output_cost"),
        start=sql.Identifier("start_date_valid"),
        end=sql.Identifier("end_date_valid"),
        table=sql.Identifier("costs"),
    )
    return query, (cost_id,)


def build_get_daily_stats_query(date: str) -> Tuple[sql.Composed, Tuple[str]]:
    """
    Builds a SQL query to get total tokens and cost for a specific date.

    :param date: Date string in YYYY-MM-DD format.
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        """
        SELECT
            COALESCE(SUM({token_input} + {token_output}), 0) as total_tokens,
            COALESCE(SUM({cost}), 0.0) as total_cost
        FROM {table}
        WHERE DATE({col_date}) = %s
    """
    ).format(
        token_input=sql.Identifier("token_input"),
        token_output=sql.Identifier("token_output"),
        cost=sql.Identifier("cost"),
        table=sql.Identifier("logs"),
        col_date=sql.Identifier("date"),
    )
    return query, (date,)


def build_get_recent_activity_query() -> Tuple[sql.Composed, Tuple[()]]:
    """
    Builds a SQL query to get the 3 most recent activities (logs and new users).
    Derives user creation date from date_valid_until (assuming 1 year validity).

    :return: Tuple of SQL query and empty parameters.
    """
    query = sql.SQL(
        """
        SELECT type, date, details FROM (
            SELECT 
                'log' as type, 
                {log_date}::timestamp as date, 
                'New Log: ' || {model} as details
            FROM {logs_table}
            
            UNION ALL
            
            SELECT 
                'user' as type, 
                ({user_date}::date - INTERVAL '1 year')::timestamp as date, 
                'New User: ' || {username} as details
            FROM {users_table}
        ) as combined_activity
        ORDER BY date DESC
        LIMIT 3
    """
    ).format(
        log_date=sql.Identifier("date"),
        model=sql.Identifier("model"),
        logs_table=sql.Identifier("logs"),
        user_date=sql.Identifier("date_valid_until"),
        username=sql.Identifier("username"),
        users_table=sql.Identifier("users"),
    )
    return query, ()


def build_insert_feedback_query(
    user_email: str,
    question: str,
    answer: str,
    feedback_value: str,
    log_id: Optional[int] = None,
    source: Optional[str] = None,
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to insert a new feedback entry.

    :param user_email: Username (email) of the user submitting the feedback.
    :param question: The question being evaluated.
    :param answer: The answer being evaluated.
    :param feedback_value: Feedback value ('positive' or 'negative').
    :param log_id: Optional reference to logs.id.
    :param source: Optional source identifier (e.g. 'agentchat', 'completion').
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        "INSERT INTO {table} "
        "({col_user}, {col_question}, {col_answer}, {col_feedback}, {col_log}, {col_source}) "
        "VALUES (%s, %s, %s, %s, %s, %s) "
        "RETURNING {col_id}"
    ).format(
        table=sql.Identifier("feedback"),
        col_user=sql.Identifier("user_email"),
        col_question=sql.Identifier("question"),
        col_answer=sql.Identifier("answer"),
        col_feedback=sql.Identifier("feedback_value"),
        col_log=sql.Identifier("log_id"),
        col_source=sql.Identifier("source"),
        col_id=sql.Identifier("id"),
    )

    params = (
        user_email,
        question,
        answer,
        feedback_value,
        log_id,
        source,
    )

    return query, params


def build_get_feedback_for_admin_query(
    limit: int,
    offset: int = 0,
    source_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to retrieve feedback entries for the admin panel with pagination.

    :param limit: Maximum number of entries.
    :param offset: Offset number.
    :param source_filter: Optional source to filter by.
    :param start_date: Start date string (YYYY-MM-DD).
    :param end_date: End date string (YYYY-MM-DD).
    :return: Tuple of SQL query and parameters.
    """
    base_query = """
        SELECT f.id, f.user_email, f.question, f.answer, f.feedback_value, f.timestamp, f.log_id, f.source, l.model
        FROM {table} f
        LEFT JOIN {logs_table} l ON f.log_id = l.id
    """

    conditions = []
    params: List[Any] = []

    if source_filter:
        conditions.append("f.{col_source} = %s")
        params.append(source_filter)

    if start_date and end_date:
        conditions.append(
            "f.timestamp::timestamp >= %s::timestamp AND f.timestamp::timestamp <= %s::timestamp"
        )
        params.append(start_date)
        params.append(end_date + " 23:59:59")

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    base_query += " ORDER BY f.timestamp DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    query = sql.SQL(base_query).format(
        table=sql.Identifier("feedback"),
        logs_table=sql.Identifier("logs"),
        col_source=sql.Identifier("source"),
    )

    return query, tuple(params)


def build_get_total_feedback_count_query(
    source_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to count total feedback entries.
    """
    base_query = "SELECT COUNT(*) FROM {table} f"

    conditions = []
    params: List[Any] = []

    if source_filter:
        conditions.append("f.{col_source} = %s")
        params.append(source_filter)

    if start_date and end_date:
        conditions.append(
            "f.timestamp::timestamp >= %s::timestamp AND f.timestamp::timestamp <= %s::timestamp"
        )
        params.append(start_date)
        params.append(end_date + " 23:59:59")

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    query = sql.SQL(base_query).format(
        table=sql.Identifier("feedback"), col_source=sql.Identifier("source")
    )

    return query, tuple(params)


def build_get_feedback_stats_query(
    source_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to retrieve feedback statistics.

    :param source_filter: Optional source to filter by.
    :return: Tuple of SQL query and parameters.
    """

    where_clause = sql.SQL("")
    conditions = []
    params: List[Any] = []

    if source_filter:
        conditions.append(
            sql.SQL("{col_source} = %s").format(col_source=sql.Identifier("source"))
        )
        params.append(source_filter)

    if start_date and end_date:
        conditions.append(
            sql.SQL(
                "timestamp::timestamp >= %s::timestamp AND timestamp::timestamp <= %s::timestamp"
            )
        )
        params.append(start_date)
        params.append(end_date + " 23:59:59")

    if conditions:
        where_clause = sql.SQL("WHERE ") + sql.SQL(" AND ").join(conditions)

    query = sql.SQL(
        """
        SELECT 
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE {col_feedback} = 'positive') as positive_count,
            COUNT(*) FILTER (WHERE {col_feedback} = 'negative') as negative_count
        FROM {table}
        {where_clause}
    """
    ).format(
        table=sql.Identifier("feedback"),
        col_feedback=sql.Identifier("feedback_value"),
        where_clause=where_clause,
    )

    return query, tuple(params)


def build_get_feedback_sources_query() -> Tuple[sql.Composed, Tuple[()]]:
    """
    Builds a SQL query to retrieve distinct feedback sources.

    :return: Tuple of SQL query and empty parameters.
    """
    query = sql.SQL(
        """
        SELECT DISTINCT {col_source} FROM {table} WHERE {col_source} IS NOT NULL ORDER BY {col_source}
    """
    ).format(table=sql.Identifier("feedback"), col_source=sql.Identifier("source"))
    return query, ()


def build_get_feedback_model_stats_query(
    source_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to retrieve feedback counts grouped by model.
    """
    base_query = """
        SELECT l.model, COUNT(f.id) as count
        FROM {table} f
        LEFT JOIN {logs_table} l ON f.log_id = l.id
    """

    conditions = []
    params: List[Any] = []

    if source_filter:
        conditions.append("f.{col_source} = %s")
        params.append(source_filter)

    if start_date and end_date:
        conditions.append(
            "f.timestamp::timestamp >= %s::timestamp AND f.timestamp::timestamp <= %s::timestamp"
        )
        params.append(start_date)
        params.append(end_date + " 23:59:59")

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    base_query += " GROUP BY l.model ORDER BY count DESC"

    query = sql.SQL(base_query).format(
        table=sql.Identifier("feedback"),
        logs_table=sql.Identifier("logs"),
        col_source=sql.Identifier("source"),
    )

    return query, tuple(params)
