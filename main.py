# === Built-in ===
import logging
import os

# Logging must be configured before any other runtime effect.
from utils.logging_config import bootstrap_logging, register_request_context_hooks

DATACHAT_RUNTIME_LOGGER = bootstrap_logging()

logger = logging.getLogger(__name__)

os.environ["MPLBACKEND"] = "Agg"

from dotenv import load_dotenv  # noqa: E402

# === Third-party ===
from flask import Flask  # noqa: E402
from flask_cors import CORS  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

# === Local modules ===
import infrastructure.database_pg as database_pg  # noqa: E402
import infrastructure.vector_store as vector_store  # noqa: E402
from config import load_config, AppConfig  # noqa: E402
from routes.system import system_bp  # noqa: E402
from routes.auth import auth_bp  # noqa: E402
from routes.users import users_bp  # noqa: E402
from routes.reporting import reporting_bp  # noqa: E402
from routes.documents import documents_bp  # noqa: E402
from routes.multimodal import multimodal_bp  # noqa: E402
from routes.ingestion import ingestion_bp  # noqa: E402
from routes.rag import rag_bp  # noqa: E402
from routes.datachat import datachat_bp  # noqa: E402
from routes.admin import admin_bp  # noqa: E402

load_dotenv()  # Load environment variables from .env file

config: AppConfig = (
    load_config()
)  # Maui runtime config, built from environment variables

database_pg.init(config)  # Init database layer config

vector_store.init(config)  # Init vector store layer config


# Initialize the Flask application
app = Flask(__name__)
register_request_context_hooks(app)  # Bind a request_id for every request
app.register_blueprint(system_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(users_bp)
app.register_blueprint(reporting_bp)
app.register_blueprint(documents_bp)
app.register_blueprint(multimodal_bp)
app.register_blueprint(ingestion_bp)
app.register_blueprint(rag_bp)
app.register_blueprint(datachat_bp)
app.register_blueprint(admin_bp)
app.config["MAUI_CONFIG"] = (
    config  # Make Maui config available to all Blueprints via current_app
)
# origins=["http://localhost:4200"]
CORS(app, expose_headers=["X-Request-ID"])

secret_key = os.environ.get("ENCRYPTION_KEY")
if not secret_key:
    raise RuntimeError("ENCRYPTION_KEY must be set in environment")
app.secret_key = secret_key

app.config["DATACHAT_RUNTIME_LOGGER"] = (
    DATACHAT_RUNTIME_LOGGER  # Make the datachat runtime logger available to all Blueprints via current_app
)

# Verify Matplotlib backend
logger.info("event=matplotlib_backend_selected backend=%s", matplotlib.get_backend())

# Removing Pandas read csv columns limitations to avoid truncated dataFrames
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
