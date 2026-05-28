# === Built-in ===
import os

os.environ["MPLBACKEND"] = "Agg"
from datetime import datetime, timedelta
from dotenv import load_dotenv
from functools import wraps

# === Third-party ===
from flask import (
    Flask,
    request,
    Response,
    render_template,
    redirect,
    url_for,
    session,
    flash,
)
from flask_cors import CORS
import pandas as pd
import matplotlib
import bcrypt
import psutil

# === Local modules ===
from services.rag_ingestion_service import process_rag_file
import infrastructure.database_pg as database_pg
import infrastructure.vector_store as vector_store
from infrastructure.database_pg import (
    get_users_for_admin,
    get_users_stats,
    get_logs_for_admin,
    get_logs_stats,
    update_user_tokens,
    get_user_by_id,
    get_all_prompts,
    get_prompt_by_id,
    add_prompt,
    update_prompt,
    delete_prompt,
    get_all_costs,
    add_cost,
    update_cost,
    delete_cost,
    get_cost_by_id,
    get_daily_stats,
    get_recent_activity,
    get_feedback_for_admin,
    get_feedback_stats,
    get_all_rag_files,
)
from utils.agent_logging import setup_agent_logger
from utils.runtime_logging import setup_datachat_runtime_logger
from config import load_config, AppConfig, PROVIDER_API_KEY_MAP
from routes.system import system_bp
from routes.auth import auth_bp
from routes.users import users_bp
from routes.reporting import reporting_bp
from routes.documents import documents_bp
from routes.multimodal import multimodal_bp
from routes.ingestion import ingestion_bp
from routes.rag import rag_bp
from routes.datachat import datachat_bp

load_dotenv()  # Load environment variables from .env file

config: AppConfig = (
    load_config()
)  # Maui runtime config, built from environment variables

database_pg.init(config)  # Init database layer config

vector_store.init(config)  # Init vector store layer config


# Initialize the Flask application
app = Flask(__name__)
app.register_blueprint(system_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(users_bp)
app.register_blueprint(reporting_bp)
app.register_blueprint(documents_bp)
app.register_blueprint(multimodal_bp)
app.register_blueprint(ingestion_bp)
app.register_blueprint(rag_bp)
app.register_blueprint(datachat_bp)
app.config["MAUI_CONFIG"] = (
    config  # Make Maui config available to all Blueprints via current_app
)
# origins=["http://localhost:4200"]
CORS(app)

secret_key = os.environ.get("ENCRYPTION_KEY")
if not secret_key:
    raise RuntimeError("ENCRYPTION_KEY must be set in environment")
app.secret_key = secret_key

# Configure the agent run logger
setup_agent_logger()

DATACHAT_RUNTIME_LOGGER = (
    setup_datachat_runtime_logger()
)  # Initialise the datachat runtime logger
app.config["DATACHAT_RUNTIME_LOGGER"] = (
    DATACHAT_RUNTIME_LOGGER  # Make the datachat runtime logger available to all Blueprints via current_app
)

# Verify Matplotlib backend
print(f"Matplotlib backend: {matplotlib.get_backend()}")

# Removing Pandas read csv columns limitations to avoid truncated dataFrames
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


# Admin authentication decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            flash("Please log in to access the admin panel", "warning")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)

    return decorated_function


# Define a route for the '/' endpoint that returns a welcome message
@app.route("/")
def welcome() -> str:
    return "Welcome to Pandino! This is the root endpoint."


# Define a route for the '/admin/costs' endpoint
@app.route("/admin/costs", methods=["GET"])
@admin_required
def admin_costs() -> str:
    costs = get_all_costs()
    return render_template("admin/costs.html", costs=costs)


# Define a route for the '/admin/costs/add' endpoint
@app.route("/admin/costs/add", methods=["POST"])
@admin_required
def admin_add_cost() -> Response:
    model = request.form.get("model")
    provider = request.form.get("provider")
    token_input_cost = float(request.form.get("token_input_cost"))
    token_output_cost = float(request.form.get("token_output_cost"))
    start_date_valid = request.form.get("start_date_valid")
    end_date_valid = request.form.get("end_date_valid")

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
    return redirect(url_for("admin_costs"))


# Define a route for the '/admin/costs/edit/<int:cost_id>' endpoint
@app.route("/admin/costs/edit/<int:cost_id>", methods=["GET", "POST"])
@admin_required
def admin_edit_cost(cost_id: int):
    if request.method == "POST":
        model = request.form.get("model")
        provider = request.form.get("provider")
        token_input_cost = float(request.form.get("token_input_cost"))
        token_output_cost = float(request.form.get("token_output_cost"))
        start_date_valid = request.form.get("start_date_valid")
        end_date_valid = request.form.get("end_date_valid")

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
        return redirect(url_for("admin_costs"))

    cost = get_cost_by_id(cost_id)
    if not cost:
        flash("Cost not found", "danger")
        return redirect(url_for("admin_costs"))
    return render_template("admin/edit_cost.html", cost=cost)


# Define a route for the '/admin/costs/delete/<int:cost_id>' endpoint
@app.route("/admin/costs/delete/<int:cost_id>", methods=["POST"])
@admin_required
def admin_delete_cost(cost_id: int):
    error = delete_cost(cost_id)
    if error:
        flash(error, "danger")
    else:
        flash("Cost deleted successfully", "success")
    return redirect(url_for("admin_costs"))


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if (
            username == config.admin.username
            and password
            and bcrypt.checkpw(password.encode("utf-8"), config.admin.password_hash)
        ):
            session["admin_logged_in"] = True
            session["admin_username"] = username
            flash("Successfully logged in!", "success")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid credentials", "danger")

    return render_template("admin/login.html")


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    session.pop("admin_username", None)
    flash("Successfully logged out", "info")
    return redirect(url_for("admin_login"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    env_vars = {
        "DATACHAT_MODEL": config.models.datachat_model,
        "DATACHAT_PROVIDER": config.models.datachat_provider,
        "PROMPT_MODEL": config.models.prompt_model,
        "PROMPT_PROVIDER": config.models.prompt_provider,
        "AUDIO_MODEL": config.models.audio_model,
        "AUDIO_PROVIDER": config.models.audio_provider,
        "COMPLETION_MODEL": config.models.completion_model,
        "COMPLETION_MODEL_PROVIDER": config.models.completion_model_provider,
        "COMPLETION_MODEL_AGENT_CHAT": config.models.completion_model_agent_chat,
        "COMPLETION_EMBEDDING_MODEL": config.models.completion_embedding_model,
        "COMPLETION_EMBEDDING_MODEL_PROVIDER": config.models.completion_embedding_model_provider,
        "WHISPER_MODEL": config.models.whisper_model,
        "VISION_PROVIDER": config.models.vision_provider,
        "VISION_MODEL": config.models.vision_model,
        "DATACHAT_TOKEN_COST": config.datachat_token_cost,
        "DATACHAT_MAX_STEPS": config.datachat.max_steps,
        "DATACHAT_RATE_LIMIT_PER_MIN": config.datachat.rate_limit_per_min,
        "DATACHAT_SESSION_TTL_MIN": config.datachat.session_ttl_min,
        "DATACHAT_LOG_LEVEL": config.datachat.log_level,
        "COMPLETION_TOKEN_COST": config.completion_token_cost,
        "PROMPT_TOKEN_COST": config.prompt_token_cost,
        "AUDIO_FORM_TOKEN_COST": config.audio_form_token_cost,
    }

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


@app.route("/admin/users")
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


@app.route("/admin/logs")
@admin_required
def admin_logs():
    try:
        # Get query parameters
        page = request.args.get("page", 1, type=int)

        # Date filter
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        search = request.args.get("search", "").strip() or None

        # Calculate default dates if not provided (for charts)
        if not start_date or not end_date:
            default_end = datetime.now()
            default_start = default_end - timedelta(days=7)
            chart_start = default_start.strftime("%Y-%m-%d")
            chart_end = default_end.strftime("%Y-%m-%d")
        else:
            chart_start = start_date
            chart_end = end_date

        logs_data = get_logs_for_admin(
            page=page,
            limit=50,
            start_date=chart_start,
            end_date=chart_end,
            search=search,
        )
        logs = logs_data["logs"]
        pagination = {
            "page": logs_data["page"],
            "total_pages": logs_data["total_pages"],
            "total_count": logs_data["total_count"],
        }

        stats = get_logs_stats(start_date=chart_start, end_date=chart_end)

        return render_template(
            "admin/logs.html",
            logs=logs,
            stats=stats,
            pagination=pagination,
            current_start_date=chart_start,
            current_end_date=chart_end,
            current_search=search or "",
        )

    except Exception as e:
        flash(f"Errore nel recupero logs: {str(e)}", "danger")
        return render_template(
            "admin/logs.html",
            logs=[],
            stats={},
            pagination={"page": 1, "total_pages": 1},
            current_start_date="",
            current_end_date="",
            current_search="",
        )


@app.route("/admin/feedback")
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


@app.route("/admin/users/<int:user_id>/edit", methods=["GET", "POST"])
@admin_required
def admin_edit_user(user_id):
    if request.method == "POST":
        try:
            new_tokens = request.form.get("tokens", type=int)

            if new_tokens is None or new_tokens < 0:
                flash("Numero di token non valido", "danger")
                return redirect(url_for("admin_users"))

            success = update_user_tokens(user_id, new_tokens)

            if success:
                flash(f"Token aggiornati con successo a {new_tokens}", "success")
            else:
                flash("Utente non trovato", "danger")

        except Exception as e:
            flash(f"Errore nell'aggiornamento: {str(e)}", "danger")

        return redirect(url_for("admin_users"))

    # GET request - show edit form
    try:
        user = get_user_by_id(user_id)
        if user:
            return render_template("admin/edit_user.html", user=user)
        else:
            flash("Utente non trovato", "danger")
            return redirect(url_for("admin_users"))
    except Exception as e:
        flash(f"Errore: {str(e)}", "danger")
        return redirect(url_for("admin_users"))


@app.route("/admin/prompts")
@admin_required
def admin_prompts():
    try:
        prompts = get_all_prompts()
        return render_template("admin/prompts.html", prompts=prompts)
    except Exception as e:
        flash(f"Errore nel recupero prompt: {str(e)}", "danger")
        return render_template("admin/prompts.html", prompts=[])


@app.route("/admin/prompts/add", methods=["POST"])
@admin_required
def admin_add_prompt():
    try:
        title = request.form.get("title")
        version = request.form.get("version", type=int)
        message = request.form.get("message")

        if not title or not version or not message:
            flash("Tutti i campi sono obbligatori", "danger")
            return redirect(url_for("admin_prompts"))

        add_prompt(title, version, message)
        flash("Prompt aggiunto con successo", "success")
    except Exception as e:
        flash(f"Errore nell'aggiunta del prompt: {str(e)}", "danger")

    return redirect(url_for("admin_prompts"))


@app.route("/admin/prompts/<int:prompt_id>/edit", methods=["GET", "POST"])
@admin_required
def admin_edit_prompt(prompt_id):
    if request.method == "POST":
        try:
            title = request.form.get("title")
            version = request.form.get("version", type=int)
            message = request.form.get("message")

            if not title or not version or not message:
                flash("Tutti i campi sono obbligatori", "danger")
                return redirect(url_for("admin_edit_prompt", prompt_id=prompt_id))

            success = update_prompt(prompt_id, title, version, message)

            if success:
                flash("Prompt aggiornato con successo", "success")
            else:
                flash("Prompt non trovato", "danger")

        except Exception as e:
            flash(f"Errore nell'aggiornamento: {str(e)}", "danger")

        return redirect(url_for("admin_prompts"))

    # GET request - show edit form
    try:
        prompt = get_prompt_by_id(prompt_id)
        if prompt:
            return render_template("admin/edit_prompt.html", prompt=prompt)
        else:
            flash("Prompt non trovato", "danger")
            return redirect(url_for("admin_prompts"))
    except Exception as e:
        flash(f"Errore: {str(e)}", "danger")
        return redirect(url_for("admin_prompts"))


@app.route("/admin/prompts/<int:prompt_id>/delete", methods=["POST"])
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

    return redirect(url_for("admin_prompts"))


@app.route("/admin/rag-files")
@admin_required
def admin_rag_files():
    try:
        rag_files = get_all_rag_files()
        return render_template("admin/rag_files.html", rag_files=rag_files)
    except Exception as e:
        flash(f"Error loading RAG files: {str(e)}", "danger")
        return render_template("admin/rag_files.html", rag_files=[])


@app.route("/admin/rag-files/upload", methods=["POST"])
@admin_required
def admin_upload_rag_file():
    file = request.files.get("file")
    namespace = request.form.get("namespace", "").strip()
    language = request.form.get("language", "").strip() or None

    if not file or not namespace:
        flash("File and namespace are required", "danger")
        return redirect(url_for("admin_rag_files"))

    url = file.filename or ""

    try:
        result = process_rag_file(
            file,
            url,
            namespace,
            language,
            whisper_model=config.models.whisper_model,
            deepinfra_api_key=config.api_keys.deepinfra_api_key,
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

    return redirect(url_for("admin_rag_files"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
