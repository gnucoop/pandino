# === Built-in ===
import os

os.environ["MPLBACKEND"] = "Agg"
from dotenv import load_dotenv

# === Third-party ===
from flask import Flask
from flask_cors import CORS
import pandas as pd
import matplotlib

# === Local modules ===
import infrastructure.database_pg as database_pg
import infrastructure.vector_store as vector_store
from utils.agent_logging import setup_agent_logger
from utils.runtime_logging import setup_datachat_runtime_logger
from config import load_config, AppConfig
from routes.system import system_bp
from routes.auth import auth_bp
from routes.users import users_bp
from routes.reporting import reporting_bp
from routes.documents import documents_bp
from routes.multimodal import multimodal_bp
from routes.ingestion import ingestion_bp
from routes.rag import rag_bp
from routes.datachat import datachat_bp
from routes.admin import admin_bp

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
app.register_blueprint(admin_bp)
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
