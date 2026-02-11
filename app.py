"""Web Gather API - Main application entry point.

This module initializes the Flask application and registers all route blueprints.
"""
from __future__ import annotations

import logging
import os

import anthropic
from dotenv import load_dotenv
from flask import Flask
from pymongo import MongoClient
from firecrawl import FirecrawlApp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("web-gather-api")

load_dotenv()

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
from core import core_bp, init_core
from util import util_bp, init_util
from routes import routes_bp
from compliance_routes import compliance_bp, init_compliance

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
# Also mount at root so POST /policy-statute-compliance works (Swagger/docs and legacy clients).
app.register_blueprint(compliance_bp, url_prefix="", name="compliance_root")

# #region agent log
try:
    import json as _json
    with open("/Users/kurthayes/Dev/AI/web-gather-api/.cursor/debug.log", "a") as _f:
        _f.write(_json.dumps({"timestamp": __import__("time").time() * 1000, "location": "app.py:blueprint_registration", "message": "Blueprints registered", "data": {"compliance_root_registered": True}, "hypothesisId": "H1"}) + "\n")
except Exception:
    pass
# #endregion

if __name__ == "__main__":
    logger.info("Starting Web Gather API server on port 7005")
    app.run(debug=True, port=7005)
