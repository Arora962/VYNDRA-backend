"""VYNDRA Flask application factory and local development entry point."""
from __future__ import annotations
import logging
import os
from flask import Flask, jsonify
from flask_cors import CORS
from config import DEFAULT_HOST, DEFAULT_PORT, MAX_REQUEST_BYTES, ROOT, cors_origins
from vyndra.database import init_db


def create_app(test_config=None) -> Flask:
    """Create and configure the Flask application without hiding import failures."""
    application = Flask(__name__)
    application.config.update(MAX_CONTENT_LENGTH=MAX_REQUEST_BYTES, JSON_SORT_KEYS=False)
    if test_config:
        application.config.update(test_config)
    logging.basicConfig(level=os.getenv("VYNDRA_LOG_LEVEL", "INFO"))
    CORS(application, resources={r"/api/*": {"origins": cors_origins()}})

    from vyndra.backend_api import bp as backend_bp
    from vyndra.chat_api import chat_bp
    from vyndra.photo_api import photo_bp
    from vyndra.social_api import social_bp
    application.register_blueprint(backend_bp)
    application.register_blueprint(chat_bp)
    application.register_blueprint(photo_bp)
    application.register_blueprint(social_bp)

    @application.get("/")
    def home():
        return jsonify({"service": "VYNDRA backend", "status": "running", "api_health": "/api/health", "docs": "docs/API_REFERENCE.md"})

    @application.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "Request exceeds the configured upload limit. Images must be no larger than 8 MB."}), 413

    @application.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "Route not found."}), 404

    @application.errorhandler(500)
    def internal_error(_error):
        application.logger.exception("Unhandled request error")
        return jsonify({"error": "Internal server error."}), 500

    with application.app_context():
        init_db()
    return application


app = create_app()

if __name__ == "__main__":
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT, debug=False)
