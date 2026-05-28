from flask import Blueprint, jsonify

system_bp = Blueprint("system", __name__)


@system_bp.route("/health")
def health():
    status = {
        "status": "ok",
    }
    return jsonify(status)


@system_bp.route("/summarize", methods=["GET"])
def summarize():
    return "The /summarize endpoint is not yet implemented.", 501


@system_bp.route("/categorize", methods=["GET"])
def categorize():
    return "The /categorize endpoint is not yet implemented.", 501


@system_bp.route("/img-comparison", methods=["GET"])
def img_comparison():
    return "The /img-comparison endpoint is not yet implemented.", 501


@system_bp.route("/")
def welcome() -> str:
    return "Welcome to Pandino! This is the root endpoint."
