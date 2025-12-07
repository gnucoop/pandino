"""
This module contains functions for safely building SQL queries.
Each function returns a composable SQL object (psycopg.sql) and its parameters.
"""

from psycopg import sql
from typing import Tuple, Any, Optional


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
        tokens=sql.Identifier("tokens"),
        table=sql.Identifier("users")
    )
    return query, ()


def build_get_logs_for_admin_query(limit: int) -> Tuple[sql.Composed, Tuple[int]]:
    """
    Builds a SQL query to retrieve logs for the admin panel.

    :param limit: Maximum number of logs to retrieve.
    :return: Tuple with SQL query and parameters.
    """
    query = sql.SQL("""
        SELECT l.id, l.user_id, u.username, l.date, l.token_input,
               l.token_output, l.cost, l.model, l.provider
        FROM {logs} l
        LEFT JOIN {users} u ON l.user_id = u.id
        ORDER BY l.date DESC
        LIMIT %s
    """).format(
        logs=sql.Identifier("logs"),
        users=sql.Identifier("users")
    )
    return query, (limit,)


def build_update_user_tokens_query(user_id: int, new_tokens: int, date_valid_until: Any) -> Tuple[sql.Composed, Tuple[Any, ...]]:
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


def build_get_total_log_stats_query() -> Tuple[sql.Composed, Tuple[()]]:
    """
    Builds a SQL query to get total log statistics.

    :return: Tuple with SQL query and empty parameter tuple.
    """
    query = sql.SQL("""
        SELECT
            COALESCE(SUM({token_input}), 0) as total_input,
            COALESCE(SUM({token_output}), 0) as total_output,
            COALESCE(SUM({cost}), 0.0) as total_cost,
            COUNT(*) as total_requests
        FROM {logs}
    """).format(
        token_input=sql.Identifier("token_input"),
        token_output=sql.Identifier("token_output"),
        cost=sql.Identifier("cost"),
        logs=sql.Identifier("logs")
    )
    return query, ()


def build_get_daily_log_stats_query() -> Tuple[sql.Composed, Tuple[()]]:
    """
    Builds a SQL query to get daily log statistics for the last 7 days.

    :return: Tuple with SQL query and empty parameter tuple.
    """
    query = sql.SQL("""
        SELECT
            DATE(date::timestamp) as day,
            SUM({token_input}) as input_tokens,
            SUM({token_output}) as output_tokens
        FROM {logs}
        WHERE date::timestamp >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY DATE(date::timestamp)
        ORDER BY day
    """).format(
        token_input=sql.Identifier("token_input"),
        token_output=sql.Identifier("token_output"),
        logs=sql.Identifier("logs")
    )
    return query, ()


def build_get_top_users_by_token_usage_query() -> Tuple[sql.Composed, Tuple[()]]:
    """
    Builds a SQL query to get top 5 users by token usage.

    :return: Tuple with SQL query and empty parameter tuple.
    """
    query = sql.SQL("""
        SELECT
            u.username,
            SUM(l.token_input + l.token_output) as total_tokens
        FROM {logs} l
        LEFT JOIN {users} u ON l.user_id = u.id
        GROUP BY u.username
        ORDER BY total_tokens DESC
        LIMIT 5
    """).format(
        logs=sql.Identifier("logs"),
        users=sql.Identifier("users")
    )
    return query, ()


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
    title: str, 
    version: Optional[int] = None
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
    query = sql.SQL("SELECT {id}, {title}, {version}, {message} FROM {table} ORDER BY {id} ASC").format(
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
    query = sql.SQL("""
        SELECT
            COALESCE(SUM({token_input} + {token_output}), 0) as total_tokens,
            COALESCE(SUM({cost}), 0.0) as total_cost
        FROM {table}
        WHERE DATE({col_date}) = %s
    """).format(
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
    query = sql.SQL("""
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
    """).format(
        log_date=sql.Identifier("date"),
        model=sql.Identifier("model"),
        logs_table=sql.Identifier("logs"),
        user_date=sql.Identifier("date_valid_until"),
        username=sql.Identifier("username"),
        users_table=sql.Identifier("users")
    )
    return query, ()


def build_query_associations_by_district_query(district: str) -> Tuple[sql.Composed, Tuple[str]]:
    """
    Builds a SQL query that returns all associations located in a given district.
    The match is case-insensitive and uses substring search (ILIKE %...%).

    :param district: District name to filter by (e.g., "Magude").
    :return: Tuple (sql.Composed query, tuple params)
    """

    # SQL template con placeholder %s per il parametro
    query = sql.SQL("""
        SELECT
            id,
            nome_associazione,
            soci_maschi,
            soci_femmine,
            descrizione,
            distretto,
            area_coltivata_ha,
            sistema_irrigazione,
            sistema_conservazione,
            sistema_processamento,
            contatto_telefonico
        FROM {table}
        WHERE {col_distretto} ILIKE %s
        ORDER BY nome_associazione ASC
    """).format(
        table=sql.Identifier("associazioni"),
        col_distretto=sql.Identifier("distretto")
    )

    # Parametro con wildcard a sinistra e destra per match flessibile
    param = (f"%{district}%",)

    return query, param


def build_query_associations_by_product_query(product: str) -> Tuple[sql.Composed, Tuple[str]]:
    """
    Builds a SQL query that returns all associations that produce a given product.
    Case-insensitive substring match (ILIKE %...%) on the 'cultura' field.

    :param product: Product name to search for (e.g., "milho").
    :return: Tuple (sql.Composed query, tuple params)
    """

    query = sql.SQL("""
        SELECT
            a.id,
            a.nome_associazione,
            a.soci_maschi,
            a.soci_femmine,
            a.descrizione,
            a.distretto,
            a.area_coltivata_ha,
            a.sistema_irrigazione,
            a.sistema_conservazione,
            a.sistema_processamento,
            a.contatto_telefonico,
            p.cultura,
            p.rendimento_estimado_kg,
            p.preco_venda_estimado_kg
        FROM {prodotti} p
        JOIN {associazioni} a
            ON p.id_associazione = a.id
        WHERE p.cultura ILIKE %s
        ORDER BY a.nome_associazione ASC;
    """).format(
        prodotti=sql.Identifier("prodotti"),
        associazioni=sql.Identifier("associazioni")
    )

    # Parametro con match parziale e case-insensitive
    params = (f"%{product}%",)

    return query, params


def build_query_product_in_district_query(product: str, district: str) -> Tuple[sql.Composed, Tuple[str, str]]:
    """
    Builds a SQL query that returns associations that produce a given product
    AND are located in a specific district.
    Both filters use case-insensitive substring matching (ILIKE %...%).

    :param product: Product name to search for (e.g., "milho").
    :param district: District name to filter by (e.g., "Magude").
    :return: Tuple (sql.Composed query, tuple params)
    """

    query = sql.SQL("""
        SELECT
            a.id,
            a.nome_associazione,
            a.soci_maschi,
            a.soci_femmine,
            a.descrizione,
            a.distretto,
            a.area_coltivata_ha,
            a.sistema_irrigazione,
            a.sistema_conservazione,
            a.sistema_processamento,
            a.contatto_telefonico,
            p.cultura,
            p.rendimento_estimado_kg,
            p.preco_venda_estimado_kg
        FROM {prodotti} p
        JOIN {associazioni} a
            ON p.id_associazione = a.id
        WHERE p.cultura ILIKE %s
          AND a.distretto ILIKE %s
        ORDER BY a.nome_associazione ASC;
    """).format(
        prodotti=sql.Identifier("prodotti"),
        associazioni=sql.Identifier("associazioni")
    )

    params = (f"%{product}%", f"%{district}%")

    return query, params


def build_query_association_details_query(name: str) -> Tuple[sql.Composed, Tuple[str]]:
    """
    Builds a SQL query that returns detailed information about a specific association.
    Match is case-insensitive and uses substring search (ILIKE %...%).

    :param name: Partial or full name of the association (e.g., "Chipene", "chip").
    :return: Tuple (sql.Composed query, tuple params)
    """

    query = sql.SQL("""
        SELECT
            id,
            nome_associazione,
            soci_maschi,
            soci_femmine,
            descrizione,
            distretto,
            area_coltivata_ha,
            sistema_irrigazione,
            sistema_conservazione,
            sistema_processamento,
            contatto_telefonico
        FROM {table}
        WHERE {col_name} ILIKE %s
        ORDER BY nome_associazione ASC;
    """).format(
        table=sql.Identifier("associazioni"),
        col_name=sql.Identifier("nome_associazione")
    )

    params = (f"%{name}%",)

    return query, params


# === Farmers Panel Queries ===

def build_create_users_associazioni_table_query() -> Tuple[Any, Tuple]:
    """
    Builds a SQL query to create the 'users_associazioni' table if it doesn't exist.

    :return: Tuple with SQL query and empty parameter tuple.
    """
    query = sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS users_associazioni (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            id_associazione INT NOT NULL,
            
            CONSTRAINT fk_ass_user
                FOREIGN KEY(id_associazione)
                REFERENCES associazioni(id)
                ON DELETE CASCADE
        );
        """
    )
    return query, ()

def build_get_farmer_by_username_query(username: str) -> Tuple[Any, Tuple]:
    """
    Builds a SQL query to select a farmer from 'users_associazioni' by username.

    :param username: The username to search for.
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        """
        SELECT id, username, password_hash, id_associazione 
        FROM users_associazioni 
        WHERE username = %s
        """
    )
    return query, (username,)

def build_get_association_by_id_query(assoc_id: int) -> Tuple[Any, Tuple]:
    """
    Builds a SQL query to select association details by ID.

    :param assoc_id: The ID of the association.
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        """
        SELECT * FROM associazioni WHERE id = %s
        """
    )
    return query, (assoc_id,)

def build_update_association_query(assoc_id: int, data: dict) -> Tuple[Any, Tuple]:
    """
    Builds a SQL query to update an association.

    :param assoc_id: The ID of the association to update.
    :param data: Dictionary of columns to update.
    :return: Tuple of SQL query and parameters.
    """
    # Columns allowed to be updated based on user request / schema
    allowed_columns = [
        'nome_associazione', 'soci_maschi', 'soci_femmine', 'descrizione',
        'distretto', 'area_coltivata_ha', 'sistema_irrigazione',
        'sistema_conservazione', 'sistema_processamento', 'contatto_telefonico'
    ]
    
    updates = []
    params = []
    
    for key, value in data.items():
        if key in allowed_columns:
            updates.append(sql.SQL("{} = %s").format(sql.Identifier(key)))
            params.append(value)
            
    if not updates:
        # Nothing to update, return a dummy query
        return sql.SQL("SELECT 1 WHERE 1=0"), ()
        
    query = sql.SQL("UPDATE associazioni SET {} WHERE id = %s").format(
        sql.SQL(', ').join(updates)
    )
    params.append(assoc_id)
    
    return query, tuple(params)

def build_get_products_by_association_query(assoc_id: int) -> Tuple[Any, Tuple]:
    """
    Builds a SQL query to select products for a specific association.

    :param assoc_id: The ID of the association.
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        """
        SELECT * FROM prodotti WHERE id_associazione = %s
        """
    )
    return query, (assoc_id,)

def build_add_product_query(data: dict) -> Tuple[Any, Tuple]:
    """
    Builds a SQL query to insert a new product.

    :param data: Dictionary containing product data.
    :return: Tuple of SQL query and parameters.
    """
    columns = ['cultura', 'rendimento_estimado_kg', 'preco_venda_estimado_kg', 'id_associazione']
    
    query = sql.SQL("INSERT INTO prodotti ({}) VALUES ({}) RETURNING id").format(
        sql.SQL(', ').join(map(sql.Identifier, columns)),
        sql.SQL(', ').join(sql.Placeholder() * len(columns))
    )
    
    params = tuple(data.get(c) for c in columns)
    return query, params

def build_update_product_query(product_id: int, data: dict) -> Tuple[Any, Tuple]:
    """
    Builds a SQL query to update a product.

    :param product_id: The ID of the product to update.
    :param data: Dictionary of columns to update.
    :return: Tuple of SQL query and parameters.
    """
    allowed_columns = ['cultura', 'rendimento_estimado_kg', 'preco_venda_estimado_kg']
    
    updates = []
    params = []
    
    for key, value in data.items():
        if key in allowed_columns:
            updates.append(sql.SQL("{} = %s").format(sql.Identifier(key)))
            params.append(value)
            
    if not updates:
         return sql.SQL("SELECT 1 WHERE 1=0"), ()

    query = sql.SQL("UPDATE prodotti SET {} WHERE id = %s").format(
        sql.SQL(', ').join(updates)
    )
    params.append(product_id)
    
    return query, tuple(params)

def build_delete_product_query(product_id: int) -> Tuple[Any, Tuple]:
    """
    Builds a SQL query to delete a product.

    :param product_id: The ID of the product to delete.
    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL("DELETE FROM prodotti WHERE id = %s")
    return query, (product_id,)


def build_get_all_farmers_query() -> Tuple[Any, Tuple]:
    """
    Builds a SQL query to select all farmers with their association name.

    :return: Tuple of SQL query and parameters.
    """
    query = sql.SQL(
        """
        SELECT u.id, u.username, u.id_associazione, a.nome_associazione, a.distretto 
        FROM users_associazioni u
        LEFT JOIN associazioni a ON u.id_associazione = a.id
        ORDER BY u.id
        """
    )
    return query, ()
