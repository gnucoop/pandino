import logging
import os
import zlib
from datetime import date, datetime, timedelta
from functools import wraps

import bcrypt
import psutil
import yaml
from dotenv import dotenv_values
from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from config import PROVIDER_API_KEY_MAP
from infrastructure.database_methods import LOG_FILTER_COLUMNS
from infrastructure.database_pg import (
    add_cost,
    add_prompt,
    delete_cost,
    delete_prompt,
    delete_rag_file,
    get_all_costs,
    get_all_prompts,
    get_rag_files_page,
    get_cost_by_id,
    get_daily_stats,
    get_feedback_for_admin,
    get_feedback_stats,
    get_log_filter_options,
    get_logs_for_admin,
    get_logs_stats,
    get_operational_events_by_request_id,
    get_prompt_by_id,
    get_recent_activity,
    get_user_by_id,
    get_users_for_admin,
    get_users_stats,
    update_cost,
    update_prompt,
    update_user_tokens,
)
from services.rag_ingestion_service import process_rag_file
from usage.attribution import (
    USAGE_POLICY_ADMIN_RAG_INGESTION,
    attribute_usage_to_policy,
)

admin_bp = Blueprint("admin", __name__)

logger = logging.getLogger(__name__)


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            flash("Please log in to access the admin panel", "warning")
            return redirect(url_for("admin.admin_login"))
        return f(*args, **kwargs)

    return decorated_function


_SAFE_VALUE_ENV_VARS = {
    "ANALYST_CHART_MAX_POINTS",
    "ANALYST_DATASET_MAX_ROWS",
    "ANALYST_CHART_MAX_STEPS",
    "ANALYST_MAX_CHARTS",
    "ANALYST_MAX_EXTRA_TASKS",
    "ANALYST_MAX_INSIGHTS",
    "ANALYST_MODEL",
    "ANALYST_OUTPUT_MAX_CHARS",
    "ANALYST_PLAN_MAX_STEPS",
    "ANALYST_PREVIEW_MAX_CHARS",
    "ANALYST_PROVIDER",
    "ANALYST_REPORT_MAX_STEPS",
    "ANALYST_REPORT_MODEL",
    "ANALYST_REPORT_PROVIDER",
    "ANALYST_SQL_MAX_ROWS",
    "ANALYST_TASK_MAX_STEPS",
    "ANALYST_TASK_RETRIES",
    "ANALYST_TOKEN_COST",
    "ASR_BASE_URL",
    "ASR_MODEL",
    "ASR_PROVIDER",
    "AUDIO_FORM_TOKEN_COST",
    "AUDIO_MODEL",
    "AUDIO_PROVIDER",
    "AUTH_GATEWAY_URL",
    "COMPARE_DOCS_MODEL",
    "COMPARE_DOCS_PROVIDER",
    "COMPARE_DOCS_TOKEN_COST",
    "COMPLETION_EMBEDDING_MODEL",
    "COMPLETION_EMBEDDING_MODEL_PROVIDER",
    "COMPLETION_MODEL",
    "COMPLETION_MODEL_AGENT_CHAT",
    "COMPLETION_MODEL_PROVIDER",
    "COMPLETION_TOKEN_COST",
    "DATACHAT_DB_HOST",
    "DATACHAT_DB_NAME",
    "DATACHAT_DB_PORT",
    "DATACHAT_DB_SCHEMA",
    "DATACHAT_DB_USER",
    "DATACHAT_ENGINE",
    "DATACHAT_LOG_LEVEL",
    "DATACHAT_MAX_STEPS",
    "DATACHAT_MODEL",
    "DATACHAT_PLOTS_DIR",
    "DATACHAT_PROVIDER",
    "DATACHAT_RATE_LIMIT_PER_MIN",
    "DATACHAT_SESSION_TTL_MIN",
    "DATACHAT_SQL_ALLOWED_TABLES",
    "DATACHAT_SQL_DENIED_TABLES",
    "DATACHAT_SQL_ENABLED",
    "DATACHAT_SQL_INCLUDE_VIEWS",
    "DATACHAT_SQL_MAX_CELL_CHARS",
    "DATACHAT_SQL_MAX_COLUMNS",
    "DATACHAT_SQL_MAX_ROWS",
    "DATACHAT_SQL_POOL_MAX_OVERFLOW",
    "DATACHAT_SQL_POOL_RECYCLE_S",
    "DATACHAT_SQL_POOL_SIZE",
    "DATACHAT_SQL_POOL_TIMEOUT_S",
    "DATACHAT_SQL_QUOTE_IDENTIFIERS",
    "DATACHAT_SQL_SCHEMA_INCLUDE_FKS",
    "DATACHAT_SQL_SCHEMA_MAX_CHARS",
    "DATACHAT_SQL_SCHEMA_PROFILE_MAX_VALUES",
    "DATACHAT_SQL_SCHEMA_PROFILE_MAX_VALUE_CHARS",
    "DATACHAT_SQL_SCHEMA_PROFILE_SAMPLE_ROWS",
    "DATACHAT_SQL_SCHEMA_PROFILE_VALUES",
    "DATACHAT_SQL_SCHEMA_TTL_S",
    "DATACHAT_SQL_STATEMENT_TIMEOUT_MS",
    "DATACHAT_TOKEN_COST",
    "INTERVIEWER_MAX_NOTES",
    "INTERVIEWER_MAX_STEPS",
    "INTERVIEWER_MAX_TURNS",
    "INTERVIEWER_MIN_QUESTIONS",
    "INTERVIEWER_MODEL",
    "INTERVIEWER_PROVIDER",
    "INTERVIEWER_TOKEN_COST",
    "LANGCHAIN_ENDPOINT",
    "LANGCHAIN_PROJECT",
    "LANGCHAIN_TRACING_V2",
    "MAUI_SCHEMA",
    "OLLAMA_BASE_URL",
    "PGPORT",
    "PGDB",
    "PGHOST",
    "PROMPT_MODEL",
    "PROMPT_PROVIDER",
    "PROMPT_TOKEN_COST",
    "RAG_DEFAULT_NAMESPACE",
    "RAG_MIN_SIM",
    "RAG_TOP_K",
    "VISION_MODEL",
    "VISION_PROVIDER",
}

_STATUS_ONLY_ENV_VARS = {
    "ADMIN_PASSWORD_HASH",
    "ANTHROPIC_API_KEY",
    "DATACHAT_DB_PASSWORD",
    "DEEPINFRA_API_KEY",
    "DEEPSEEK_API_KEY",
    "ENCRYPTION_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "LANGCHAIN_API_KEY",
    "MISTRAL_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "PANDASAI_API_KEY",
    "PGPWD",
    "PINECONE_API_KEY",
    "STRIPE_SK_KEY",
    "TOGETHER_API_KEY",
    "X_AUTH_TOKEN",
}

_DASHBOARD_ENV_ALLOWLIST = _SAFE_VALUE_ENV_VARS | _STATUS_ONLY_ENV_VARS


def _collect_env_vars(root_path: str) -> dict:
    """Env-var name→value map for the dashboard.

    Reads from the .env file when present (dev); otherwise falls back to live
    os.environ (Docker/prod, where --env-file injects vars without shipping the
    file). Only explicitly allowlisted variables are emitted. Sensitive
    allowlisted variables expose configuration status only.
    """
    env_path = os.path.join(root_path, ".env")
    if os.path.exists(env_path):
        raw = dict(dotenv_values(env_path))
    else:
        raw = {
            key: os.environ.get(key)
            for key in _DASHBOARD_ENV_ALLOWLIST
            if key in os.environ
        }

    display = {}
    for key, value in raw.items():
        if key in _STATUS_ONLY_ENV_VARS:
            display[key] = "configured" if value else "not set"
        elif key in _SAFE_VALUE_ENV_VARS:
            display[key] = value if value not in (None, "") else "not set"
    return display


# Define a route for the '/admin/costs' endpoint
@admin_bp.route("/admin/costs", methods=["GET"])
@admin_required
def admin_costs() -> str:
    costs = get_all_costs()
    return render_template("admin/costs.html", costs=costs)


# Define a route for the '/admin/costs/add' endpoint
@admin_bp.route("/admin/costs/add", methods=["POST"])
@admin_required
def admin_add_cost():
    model = request.form.get("model")
    provider = request.form.get("provider")
    token_input_cost = request.form.get("token_input_cost")
    token_output_cost = request.form.get("token_output_cost")
    start_date_valid = request.form.get("start_date_valid")
    end_date_valid = request.form.get("end_date_valid")

    if (
        not model
        or not provider
        or not token_input_cost
        or not token_output_cost
        or not start_date_valid
        or not end_date_valid
    ):
        flash("All fields are required", "danger")
        return redirect(url_for("admin.admin_costs"))

    token_input_cost = float(token_input_cost)
    token_output_cost = float(token_output_cost)

    error = add_cost(
        model,
        provider,
        token_input_cost,
        token_output_cost,
        start_date_valid,
        end_date_valid,
    )
    if error:
        flash(error, "danger")
    else:
        flash("Cost added successfully", "success")
    return redirect(url_for("admin.admin_costs"))


# Define a route for the '/admin/costs/edit/<int:cost_id>' endpoint
@admin_bp.route("/admin/costs/edit/<int:cost_id>", methods=["GET", "POST"])
@admin_required
def admin_edit_cost(cost_id: int):
    if request.method == "POST":
        model = request.form.get("model")
        provider = request.form.get("provider")
        token_input_cost = request.form.get("token_input_cost")
        token_output_cost = request.form.get("token_output_cost")
        start_date_valid = request.form.get("start_date_valid")
        end_date_valid = request.form.get("end_date_valid")

        if (
            not model
            or not provider
            or not token_input_cost
            or not token_output_cost
            or not start_date_valid
            or not end_date_valid
        ):
            flash("All fields are required", "danger")
            return redirect(url_for("admin.admin_costs"))

        token_input_cost = float(token_input_cost)
        token_output_cost = float(token_output_cost)

        error = update_cost(
            cost_id,
            model,
            provider,
            token_input_cost,
            token_output_cost,
            start_date_valid,
            end_date_valid,
        )
        if error:
            flash(error, "danger")
        else:
            flash("Cost updated successfully", "success")
        return redirect(url_for("admin.admin_costs"))

    cost = get_cost_by_id(cost_id)
    if not cost:
        flash("Cost not found", "danger")
        return redirect(url_for("admin.admin_costs"))
    return render_template("admin/edit_cost.html", cost=cost)


# Define a route for the '/admin/costs/delete/<int:cost_id>' endpoint
@admin_bp.route("/admin/costs/delete/<int:cost_id>", methods=["POST"])
@admin_required
def admin_delete_cost(cost_id: int):
    error = delete_cost(cost_id)
    if error:
        flash(error, "danger")
    else:
        flash("Cost deleted successfully", "success")
    return redirect(url_for("admin.admin_costs"))


@admin_bp.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        config = current_app.config["MAUI_CONFIG"]

        if (
            username == config.admin.username
            and password
            and bcrypt.checkpw(password.encode("utf-8"), config.admin.password_hash)
        ):
            session["admin_logged_in"] = True
            session["admin_username"] = username
            flash("Successfully logged in!", "success")
            return redirect(url_for("admin.admin_dashboard"))
        else:
            flash("Invalid credentials", "danger")

    return render_template("admin/login.html")


@admin_bp.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    session.pop("admin_username", None)
    flash("Successfully logged out", "info")
    return redirect(url_for("admin.admin_login"))


@admin_bp.route("/admin/api-docs", methods=["GET"])
@admin_required
def admin_api_docs() -> str:
    """Render the Swagger UI page for the Pandino HTTP API."""
    return render_template("admin/api_docs.html")


@admin_bp.route("/admin/openapi.json", methods=["GET"])
@admin_required
def admin_openapi_spec():
    """Serve the hand-maintained OpenAPI spec (project_docs/openapi.yaml) as JSON.

    Served behind admin_required so the spec is only reachable by logged-in
    admins, and returned as JSON to avoid YAML content-type quirks in Swagger UI.
    """
    spec_path = os.path.join(current_app.root_path, "project_docs", "openapi.yaml")
    with open(spec_path, "r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    return jsonify(spec)


@admin_bp.route("/admin")
@admin_required
def admin_dashboard():
    env_vars = _collect_env_vars(current_app.root_path)

    try:
        stats_data = get_users_stats()

        today = datetime.now().strftime("%Y-%m-%d")
        daily_stats = get_daily_stats(today)
        recent_activity = get_recent_activity()

        stats = {
            "total_users": stats_data["total_users"],
            "active_sessions": stats_data[
                "total_tokens"
            ],  # Keeping this for now, but will replace in template
            "daily_tokens": daily_stats["total_tokens"],
            "daily_cost": daily_stats["total_cost"],
            "total_orders": 0,
            "recent_activity": recent_activity,
            "db_connected": True,  # If we got here, DB is connected
            "cpu_percent": psutil.cpu_percent(interval=0.5),  # utilizzo CPU
            "memory": {
                "total": psutil.virtual_memory().total,
                "used": psutil.virtual_memory().used,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
            },
        }
        return render_template("admin/dashboard.html", stats=stats, env_vars=env_vars)

    except Exception as e:
        flash(f"Errore nel caricamento dashboard: {str(e)}", "danger")
        stats = {
            "total_users": 0,
            "active_sessions": 0,
            "total_orders": 0,
            "db_connected": False,  # DB connection failed
            "recent_activity": [],
            "daily_tokens": 0,
            "daily_cost": 0.0,
            "cpu_percent": 0,
            "memory": {"total": 0, "used": 0, "available": 0, "percent": 0},
        }

        return render_template("admin/dashboard.html", stats=stats, env_vars=env_vars)


@admin_bp.route("/admin/users")
@admin_required
def admin_users():
    try:
        page = request.args.get("page", 1, type=int)
        search = request.args.get("search", "").strip() or None
        users_data = get_users_for_admin(page=page, limit=50, search=search)
        users = users_data["users"]
        pagination = {
            "page": users_data["page"],
            "total_pages": users_data["total_pages"],
            "total_count": users_data["total_count"],
        }
        return render_template(
            "admin/users.html",
            users=users,
            pagination=pagination,
            current_search=search or "",
        )

    except Exception as e:
        flash(f"Errore nel recupero utenti: {str(e)}", "danger")
        return render_template(
            "admin/users.html",
            users=[],
            pagination={"page": 1, "total_pages": 1},
            current_search="",
        )


LOGS_DEFAULT_PER_PAGE = 50
LOGS_PER_PAGE_OPTIONS = (25, 50, 100, 200)
LOGS_DEFAULT_DAYS = 7
LOGS_MAX_FILTER_LENGTH = 200
LOGS_FILTER_LABELS = {
    "user": "User",
    "service": "Service",
    "source": "Source",
    "model": "Model",
    "provider": "Provider",
    "search": "Search",
}

# Soft Bootstrap 5.3 colour pairs for value badges (service, source, ...).
_BADGE_PALETTE = (
    "primary",
    "success",
    "info",
    "warning",
    "danger",
    "secondary",
    "dark",
)


def badge_class(value) -> str:
    """Stable badge colour for a value, so e.g. a service keeps its colour."""
    if value in (None, "", "N/A", "Unknown"):
        return "bg-light text-muted border"
    color = _BADGE_PALETTE[zlib.crc32(str(value).encode()) % len(_BADGE_PALETTE)]
    return f"bg-{color}-subtle text-{color}-emphasis border border-{color}-subtle"


def _parse_iso_date(value):
    try:
        return datetime.strptime((value or "").strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _clean_filter_value(value):
    value = (value or "").strip()
    if not value or len(value) > LOGS_MAX_FILTER_LENGTH:
        return None
    return value


def _logs_quick_ranges(today: date):
    return [
        ("today", "Today", today, today),
        ("7d", "Last 7 days", today - timedelta(days=LOGS_DEFAULT_DAYS - 1), today),
        ("30d", "Last 30 days", today - timedelta(days=29), today),
        ("month", "This month", today.replace(day=1), today),
    ]


def _parse_logs_query(args, today: date) -> dict:
    """
    Validate the Usage page query string. Anything malformed falls back to
    its default rather than reaching SQL: dates must be YYYY-MM-DD (missing or
    invalid -> the last LOGS_DEFAULT_DAYS days; reversed -> swapped), ``user``
    must be a positive integer id, text filters are trimmed and capped at
    LOGS_MAX_FILTER_LENGTH, ``page`` >= 1 and ``per_page`` one of
    LOGS_PER_PAGE_OPTIONS.
    """
    start = _parse_iso_date(args.get("start_date"))
    end = _parse_iso_date(args.get("end_date"))
    dates_defaulted = False
    if start and not end:
        end = max(start, today)
    elif end and not start:
        start = end - timedelta(days=LOGS_DEFAULT_DAYS - 1)
    elif not start and not end:
        start = today - timedelta(days=LOGS_DEFAULT_DAYS - 1)
        end = today
        dates_defaulted = True
    if start > end:
        start, end = end, start

    filters = {}
    user_id = args.get("user", type=int)
    if user_id is not None and user_id > 0:
        filters["user_id"] = user_id
    for column in LOG_FILTER_COLUMNS:
        value = _clean_filter_value(args.get(column))
        if value:
            filters[column] = value

    page = args.get("page", type=int)
    if page is None or page < 1:
        page = 1
    per_page = args.get("per_page", type=int)
    if per_page not in LOGS_PER_PAGE_OPTIONS:
        per_page = LOGS_DEFAULT_PER_PAGE

    return {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "dates_defaulted": dates_defaulted,
        "search": _clean_filter_value(args.get("search")),
        "filters": filters,
        "page": page,
        "per_page": per_page,
    }


def _logs_filter_args(q: dict) -> dict:
    """URL parameters reproducing the current view (without ``page``)."""
    args = {}
    if not q["dates_defaulted"]:
        args["start_date"] = q["start_date"]
        args["end_date"] = q["end_date"]
    if q["search"]:
        args["search"] = q["search"]
    if "user_id" in q["filters"]:
        args["user"] = q["filters"]["user_id"]
    for column in LOG_FILTER_COLUMNS:
        if column in q["filters"]:
            args[column] = q["filters"][column]
    if q["per_page"] != LOGS_DEFAULT_PER_PAGE:
        args["per_page"] = q["per_page"]
    return args


def _empty_log_filter_options() -> dict:
    options = {"users": []}
    options.update({column: [] for column in LOG_FILTER_COLUMNS})
    return options


def _include_selected_options(options: dict, filters: dict) -> dict:
    """Keep a selected value listed even when it has no rows in the range."""
    for column in LOG_FILTER_COLUMNS:
        value = filters.get(column)
        if value and value not in options[column]:
            options[column] = [value] + options[column]
    user_id = filters.get("user_id")
    if user_id is not None and all(u["id"] != user_id for u in options["users"]):
        options["users"] = [{"id": user_id, "username": f"#{user_id}"}] + options[
            "users"
        ]
    return options


@admin_bp.route("/admin/logs")
@admin_required
def admin_logs():
    today = date.today()
    q = _parse_logs_query(request.args, today)
    filter_args = _logs_filter_args(q)
    query_kwargs = {
        "start_date": q["start_date"],
        "end_date": q["end_date"],
        "search": q["search"],
        "filters": q["filters"],
    }

    try:
        logs_data = get_logs_for_admin(
            page=q["page"], limit=q["per_page"], **query_kwargs
        )
        stats = get_logs_stats(**query_kwargs)
        options = get_log_filter_options(q["start_date"], q["end_date"])
    except Exception as e:
        flash(f"Errore nel recupero logs: {str(e)}", "danger")
        logs_data = {"logs": [], "page": 1, "total_pages": 1, "total_count": 0}
        stats = {}
        options = _empty_log_filter_options()

    options = _include_selected_options(options, q["filters"])
    logs = logs_data["logs"]
    page = logs_data["page"]
    pagination = {
        "page": page,
        "total_pages": logs_data["total_pages"],
        "total_count": logs_data["total_count"],
        "per_page": q["per_page"],
        "first_index": (page - 1) * q["per_page"] + 1 if logs else 0,
        "last_index": (page - 1) * q["per_page"] + len(logs),
    }

    usernames = {u["id"]: u["username"] for u in options["users"]}
    active_filters = []
    for key in ("user", *LOG_FILTER_COLUMNS, "search"):
        if key not in filter_args:
            continue
        value = filter_args[key]
        remaining = {k: v for k, v in filter_args.items() if k != key}
        active_filters.append(
            {
                "key": key,
                "label": LOGS_FILTER_LABELS[key],
                "value": usernames.get(value, value) if key == "user" else value,
                "remove_url": url_for("admin.admin_logs", **remaining),
            }
        )

    quick_ranges = []
    for key, label, start, end in _logs_quick_ranges(today):
        args = dict(filter_args, start_date=start.isoformat(), end_date=end.isoformat())
        quick_ranges.append(
            {
                "key": key,
                "label": label,
                "url": url_for("admin.admin_logs", **args),
                "active": start.isoformat() == q["start_date"]
                and end.isoformat() == q["end_date"],
            }
        )

    return render_template(
        "admin/logs.html",
        logs=logs,
        stats=stats,
        pagination=pagination,
        current_start_date=q["start_date"],
        current_end_date=q["end_date"],
        current_search=q["search"] or "",
        dates_defaulted=q["dates_defaulted"],
        current_filters={
            "user": q["filters"].get("user_id"),
            **{column: q["filters"].get(column, "") for column in LOG_FILTER_COLUMNS},
        },
        filter_options=options,
        filter_args=filter_args,
        active_filters=active_filters,
        quick_ranges=quick_ranges,
        per_page_options=LOGS_PER_PAGE_OPTIONS,
        badge_class=badge_class,
    )


@admin_bp.route("/admin/logs/<request_id>/operational", methods=["GET"])
@admin_required
def admin_operational_timeline(request_id: str):
    """Render the Operational event timeline correlated to one request_id.

    Read-only drill-down reached from a Usage row. Usage coverage and
    Operational coverage are intentionally non-symmetric, so a valid
    request_id legitimately resolves to an empty timeline, and a request_id
    with no Usage row at all is still a valid correlation key - no Usage
    lookup is performed here.

    A genuine Operational read failure is contained in this page: it renders
    the timeline page in its failure state and emits one runtime log. The
    Usage page is a separate request and stays unaffected, and the failure is
    never converted into an empty timeline.
    """
    request_id = (request_id or "").strip()
    if not request_id or request_id == "N/A":
        flash("Invalid request ID", "danger")
        return redirect(url_for("admin.admin_logs"))

    events = []
    read_failed = False
    try:
        events = get_operational_events_by_request_id(request_id)
    except Exception as e:
        read_failed = True
        logger.exception(
            "event=admin_operational_timeline_read_failed request_id=%s error_type=%s",
            request_id,
            type(e).__name__,
        )

    return render_template(
        "admin/operational_timeline.html",
        request_id=request_id,
        events=events,
        read_failed=read_failed,
    )


@admin_bp.route("/admin/feedback")
@admin_required
def admin_feedback():
    try:
        source_filter = request.args.get("source")
        if source_filter == "all":
            source_filter = None

        page = request.args.get("page", 1, type=int)

        # Date filter
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        # Calculate default dates if not provided
        if not start_date or not end_date:
            default_end = datetime.now()
            default_start = default_end - timedelta(
                days=30
            )  # Default to last 30 days for feedback
            chart_start = default_start.strftime("%Y-%m-%d")
            chart_end = default_end.strftime("%Y-%m-%d")
        else:
            chart_start = start_date
            chart_end = end_date

        feedback_data = get_feedback_for_admin(
            source_filter,
            page=page,
            limit=20,
            start_date=chart_start,
            end_date=chart_end,
        )
        feedbacks = feedback_data["feedbacks"]
        pagination = {
            "page": feedback_data["page"],
            "total_pages": feedback_data["total_pages"],
            "total_count": feedback_data["total_count"],
        }

        stats = get_feedback_stats(
            source_filter, start_date=chart_start, end_date=chart_end
        )

        return render_template(
            "admin/feedback.html",
            feedbacks=feedbacks,
            stats=stats,
            current_filter=source_filter,
            pagination=pagination,
            current_start_date=chart_start,
            current_end_date=chart_end,
        )
    except Exception as e:
        flash(f"Errore nel recupero feedback: {str(e)}", "danger")
        return render_template(
            "admin/feedback.html",
            feedbacks=[],
            stats={},
            pagination={"page": 1, "total_pages": 1},
            current_start_date="",
            current_end_date="",
        )


@admin_bp.route("/admin/users/<int:user_id>/edit", methods=["GET", "POST"])
@admin_required
def admin_edit_user(user_id):
    if request.method == "POST":
        try:
            new_tokens = request.form.get("tokens", type=int)

            if new_tokens is None or new_tokens < 0:
                flash("Numero di token non valido", "danger")
                return redirect(url_for("admin.admin_users"))

            success = update_user_tokens(user_id, new_tokens)

            if success:
                flash(f"Token aggiornati con successo a {new_tokens}", "success")
            else:
                flash("Utente non trovato", "danger")

        except Exception as e:
            flash(f"Errore nell'aggiornamento: {str(e)}", "danger")

        return redirect(url_for("admin.admin_users"))

    # GET request - show edit form
    try:
        user = get_user_by_id(user_id)
        if user:
            return render_template("admin/edit_user.html", user=user)
        else:
            flash("Utente non trovato", "danger")
            return redirect(url_for("admin.admin_users"))
    except Exception as e:
        flash(f"Errore: {str(e)}", "danger")
        return redirect(url_for("admin.admin_users"))


@admin_bp.route("/admin/prompts")
@admin_required
def admin_prompts():
    try:
        prompts = get_all_prompts()
        return render_template("admin/prompts.html", prompts=prompts)
    except Exception as e:
        flash(f"Errore nel recupero prompt: {str(e)}", "danger")
        return render_template("admin/prompts.html", prompts=[])


@admin_bp.route("/admin/prompts/add", methods=["POST"])
@admin_required
def admin_add_prompt():
    try:
        title = request.form.get("title")
        version = request.form.get("version", type=int)
        message = request.form.get("message")

        if not title or not version or not message:
            flash("Tutti i campi sono obbligatori", "danger")
            return redirect(url_for("admin.admin_prompts"))

        add_prompt(title, version, message)
        flash("Prompt aggiunto con successo", "success")
    except Exception as e:
        flash(f"Errore nell'aggiunta del prompt: {str(e)}", "danger")

    return redirect(url_for("admin.admin_prompts"))


@admin_bp.route("/admin/prompts/<int:prompt_id>/edit", methods=["GET", "POST"])
@admin_required
def admin_edit_prompt(prompt_id):
    if request.method == "POST":
        try:
            title = request.form.get("title")
            version = request.form.get("version", type=int)
            message = request.form.get("message")

            if not title or not version or not message:
                flash("Tutti i campi sono obbligatori", "danger")
                return redirect(url_for("admin.admin_edit_prompt", prompt_id=prompt_id))

            success = update_prompt(prompt_id, title, version, message)

            if success:
                flash("Prompt aggiornato con successo", "success")
            else:
                flash("Prompt non trovato", "danger")

        except Exception as e:
            flash(f"Errore nell'aggiornamento: {str(e)}", "danger")

        return redirect(url_for("admin.admin_prompts"))

    # GET request - show edit form
    try:
        prompt = get_prompt_by_id(prompt_id)
        if prompt:
            return render_template("admin/edit_prompt.html", prompt=prompt)
        else:
            flash("Prompt non trovato", "danger")
            return redirect(url_for("admin.admin_prompts"))
    except Exception as e:
        flash(f"Errore: {str(e)}", "danger")
        return redirect(url_for("admin.admin_prompts"))


@admin_bp.route("/admin/prompts/<int:prompt_id>/delete", methods=["POST"])
@admin_required
def admin_delete_prompt(prompt_id):
    try:
        success = delete_prompt(prompt_id)
        if success:
            flash("Prompt eliminato con successo", "success")
        else:
            flash("Prompt non trovato", "danger")
    except Exception as e:
        flash(f"Errore nell'eliminazione: {str(e)}", "danger")

    return redirect(url_for("admin.admin_prompts"))


RAG_FILES_DEFAULT_PER_PAGE = 25
RAG_FILES_PER_PAGE_OPTIONS = (10, 25, 50, 100)


def _parse_rag_files_paging(source) -> tuple[int, int]:
    """
    Read ``page`` / ``per_page`` from a request mapping (args or form).

    Non-numeric or out-of-range values fall back to the defaults: page 1 and
    RAG_FILES_DEFAULT_PER_PAGE. ``per_page`` must be one of
    RAG_FILES_PER_PAGE_OPTIONS. A page past the end is clamped later, once the
    total is known.
    """
    page = source.get("page", type=int)
    if page is None or page < 1:
        page = 1
    per_page = source.get("per_page", type=int)
    if per_page not in RAG_FILES_PER_PAGE_OPTIONS:
        per_page = RAG_FILES_DEFAULT_PER_PAGE
    return page, per_page


def _rag_files_redirect(page: int = 1, per_page: int = RAG_FILES_DEFAULT_PER_PAGE):
    params = {}
    if page != 1:
        params["page"] = page
    if per_page != RAG_FILES_DEFAULT_PER_PAGE:
        params["per_page"] = per_page
    return redirect(url_for("admin.admin_rag_files", **params))


@admin_bp.route("/admin/rag-files")
@admin_required
def admin_rag_files():
    page, per_page = _parse_rag_files_paging(request.args)
    try:
        data = get_rag_files_page(page=page, per_page=per_page)
    except Exception as e:
        flash(f"Error loading RAG files: {str(e)}", "danger")
        data = {
            "rag_files": [],
            "page": 1,
            "per_page": per_page,
            "total_count": 0,
            "total_pages": 1,
        }

    pagination = {
        "page": data["page"],
        "per_page": data["per_page"],
        "total_count": data["total_count"],
        "total_pages": data["total_pages"],
        "first_index": (data["page"] - 1) * data["per_page"] + 1
        if data["rag_files"]
        else 0,
        "last_index": (data["page"] - 1) * data["per_page"] + len(data["rag_files"]),
    }
    return render_template(
        "admin/rag_files.html",
        rag_files=data["rag_files"],
        pagination=pagination,
        per_page_options=RAG_FILES_PER_PAGE_OPTIONS,
    )


@admin_bp.route("/admin/rag-files/upload", methods=["POST"])
@admin_required
def admin_upload_rag_file():
    config = current_app.config["MAUI_CONFIG"]
    file = request.files.get("file")
    namespace = request.form.get("namespace", "").strip()
    language = request.form.get("language", "").strip() or None
    # New files sort first, so land on page 1 with the viewer's page size.
    _page, per_page = _parse_rag_files_paging(request.form)

    if not file or not namespace:
        flash("File and namespace are required", "danger")
        return _rag_files_redirect(per_page=per_page)

    # Declared after validation - a request that cannot embed anything
    # resolves no identity - and before every line of the ingestion attempt,
    # so it precedes any provider work that could emit embedding
    # contributions. The admin session authenticates the operator, which is
    # not the accounting principal: RAG ingestion has no end user, so the
    # intent is a technical accounting policy. Which provisioned identity
    # serves that policy, and the ratified source=None rule, belong to the
    # Usage boundary.
    attribute_usage_to_policy(policy=USAGE_POLICY_ADMIN_RAG_INGESTION)

    url = file.filename or ""

    asr_provider = config.models.asr_provider
    asr_api_key = os.getenv(
        PROVIDER_API_KEY_MAP.get(asr_provider or "", "")
    )
    asr_base_url = config.models.asr_base_url

    try:
        result = process_rag_file(
            file,
            url,
            namespace,
            language,
            asr_model=config.models.asr_model,
            asr_provider=asr_provider,
            asr_api_key=asr_api_key,
            asr_base_url=asr_base_url,
            vision_provider=config.models.vision_provider,
            vision_model=config.models.vision_model,
            embedding_provider=config.models.completion_embedding_model_provider,
            embedding_model=config.models.completion_embedding_model,
            vision_api_key=os.getenv(
                PROVIDER_API_KEY_MAP.get(config.models.vision_provider or "", "")
            ),
            embedding_api_key=os.getenv(
                PROVIDER_API_KEY_MAP.get(
                    config.models.completion_embedding_model_provider or "", ""
                )
            ),
        )

        if result.chunk_count > 0:
            flash(
                f"File indexed successfully ({result.chunk_count} chunks)",
                "success",
            )
        else:
            flash("File was empty, nothing indexed", "warning")
    except Exception as e:
        flash(f"Error processing file: {str(e)}", "danger")

    return _rag_files_redirect(per_page=per_page)


@admin_bp.route("/admin/rag-files/delete", methods=["POST"])
@admin_required
def admin_delete_rag_file():
    file_id = request.form.get("file_id")
    namespace = request.form.get("namespace")
    # Return to the page the delete came from; the listing clamps it if that
    # page no longer exists after the delete.
    page, per_page = _parse_rag_files_paging(request.form)

    if not file_id or not namespace:
        flash("Missing file id or namespace", "danger")
        return _rag_files_redirect(page, per_page)

    try:
        result = delete_rag_file(file_id, namespace)
        if result["row_deleted"]:
            flash(
                f"RAG file deleted ({result['chunks_deleted']} chunks removed)",
                "success",
            )
        else:
            flash("RAG file not found", "warning")
    except Exception as e:
        flash(f"Error deleting RAG file: {str(e)}", "danger")

    return _rag_files_redirect(page, per_page)
