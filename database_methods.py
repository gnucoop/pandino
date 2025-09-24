"""
This module contains functions for safely building SQL queries.
Each function returns a composable SQL object (psycopg.sql) and its parameters.
"""

from psycopg import sql
from typing import Tuple, Any


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
    username: str, encrypted_api_key: str, date_valid_until: str
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to insert a new user into the 'users' table.

    :param username: Unique username of the user.
    :param encrypted_api_key: API key already encrypted as string.
    :param date_valid_until: Expiration date in ISO format.
    :return: Tuple with SQL query and parameters.
    """
    query = sql.SQL(
        "INSERT INTO {table} ({col_username}, {col_api_key}, {col_date}) "
        "VALUES (%s, %s, %s)"
    ).format(
        table=sql.Identifier("users"),
        col_username=sql.Identifier("username"),
        col_api_key=sql.Identifier("api_key"),
        col_date=sql.Identifier("date_valid_until"),
    )
    params = (username, encrypted_api_key, date_valid_until)
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


def build_list_users_query() -> Tuple[sql.Composed, Tuple[()]]:
    """
    Builds a SQL query to retrieve all users with selected columns from the 'users' table.

    :return: Tuple with SQL query and empty parameter tuple.
    """
    query = sql.SQL(
        "SELECT {id}, {username}, {api_key}, {date_valid_until}, {tokens} FROM {table}"
    ).format(
        id=sql.Identifier("id"),
        username=sql.Identifier("username"),
        api_key=sql.Identifier("api_key"),
        date_valid_until=sql.Identifier("date_valid_until"),
        tokens=sql.Identifier("tokens"),
        table=sql.Identifier("users"),
    )
    return query, ()


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
    Builds a SQL query to retrieve API keys and expiration dates for a given user.

    :param username: The username to search for.
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        "SELECT {col_key}, {col_date} FROM {table} WHERE {col_user} = %s"
    ).format(
        col_key=sql.Identifier("api_key"),
        col_date=sql.Identifier("date_valid_until"),
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
) -> Tuple[sql.Composed, Tuple[Any, ...]]:
    """
    Builds a SQL query to insert a new usage log into the 'logs' table.

    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        "INSERT INTO {table} ({col_date}, {col_user}, {col_in}, {col_out}, {col_cost}, {col_model}, {col_provider}) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)"
    ).format(
        table=sql.Identifier("logs"),
        col_date=sql.Identifier("date"),
        col_user=sql.Identifier("user_id"),
        col_in=sql.Identifier("token_input"),
        col_out=sql.Identifier("token_output"),
        col_cost=sql.Identifier("cost"),
        col_model=sql.Identifier("model"),
        col_provider=sql.Identifier("provider"),
    )
    params = (date, user_id, token_input, token_output, cost, model, provider)
    return query, params
