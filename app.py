"""Web Gather API - Main application entry point.

This module initializes the Flask application and registers all route blueprints.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, request
from pymongo import MongoClient
from firecrawl import FirecrawlApp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("web-gather-api")

# Load .env from project root (works when cwd differs, e.g. PythonAnywhere WSGI)
load_dotenv(Path(__file__).resolve().parent / ".env")

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not FIRECRAWL_API_KEY:
    raise RuntimeError("FIRECRAWL_API_KEY is not set")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set")

logger.info("Initializing Flask application")
app = Flask(__name__)

logger.info("Connecting to MongoDB at %s", MONGODB_URI.split("@")[-1] if "@" in MONGODB_URI else "localhost")
mongo_client = MongoClient(MONGODB_URI)

logger.info("Initializing Firecrawl client")
firecrawl_client = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

logger.info("Initializing Anthropic client")
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Import and initialize modules
from db import db_bp, init_db
from core import core_bp, init_core, init_index_job
from util import util_bp, init_util
from routes import routes_bp
from compliance_routes import compliance_bp, init_compliance
from compliance_routes_v2 import compliance_v2_bp
from compliance_routes_v3 import compliance_v3_bp

# Initialize modules with required clients
init_db(mongo_client)
init_core(mongo_client, firecrawl_client, anthropic_client)
init_util(mongo_client)
init_compliance(mongo_client)

# Register blueprints
logger.info("Registering route blueprints")
app.register_blueprint(db_bp)
app.register_blueprint(core_bp)
app.register_blueprint(util_bp)
app.register_blueprint(routes_bp)
app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
app.register_blueprint(compliance_v2_bp, url_prefix="/api/v2/compliance")
app.register_blueprint(compliance_v3_bp, url_prefix="/api/v3/compliance")
# Also mount at root so POST /policy-statute-compliance works (Swagger/docs and legacy clients).
app.register_blueprint(compliance_bp, url_prefix="", name="compliance_root")

# Initialize index job service (requires app for test client in background workflow)
init_index_job(app, mongo_client)


@app.get("/")
def index():
    """Redirect root to docs."""
    return redirect("/docs", code=302)


@app.errorhandler(404)
def redirect_404_to_docs(_exc):
    """Redirect browser GET requests for non-API paths to docs."""
    if request.method == "GET" and not request.path.startswith("/api"):
        return redirect("/docs", code=302)
    return jsonify({"error": "Not found"}), 404


@app.before_request
def log_request_params():
    """Log all API parameters for every request."""
    params = {"method": request.method, "path": request.path}
    if request.args:
        params["query"] = dict(request.args)
    if request.method in ("POST", "PUT", "PATCH") and request.is_json:
        body = request.get_json(silent=True)
        if body is not None:
            params["body"] = body
    elif request.form:
        params["form"] = dict(request.form)
    logger.info("API request: %s", json.dumps(params, default=str))


if __name__ == "__main__":
    logger.info("Starting Web Gather API server on port 7005")
    app.run(debug=True, port=7005)
