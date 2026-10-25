"""VYNDRA Flask backend entry point. Run with: python app.py"""
from pathlib import Path
import os
from flask import Flask, jsonify
from flask_cors import CORS
from vyndra.database import init_db
from vyndra.backend_api import bp
ROOT=Path(__file__).resolve().parent
app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=8*1024*1024
# For local development only. Restrict origins via VYNDRA_CORS_ORIGINS before deployment.
origins=[x.strip() for x in os.getenv('VYNDRA_CORS_ORIGINS','http://localhost:8081,http://127.0.0.1:8081').split(',') if x.strip()]
CORS(app, resources={r'/api/*': {'origins': origins}})
app.register_blueprint(bp)
try:
 from vyndra.chat_api import chat_bp
 app.register_blueprint(chat_bp)
except Exception as exc:
 app.logger.warning('Optional Part 4 chat blueprint not registered: %s', exc)
try:
 from vyndra.social_api import social_bp
 app.register_blueprint(social_bp)
except Exception as exc:
 app.logger.warning('Optional Part 5 social blueprint not registered: %s', exc)
@app.get('/')
def home():
 return jsonify({'service':'VYNDRA backend','status':'running','api_health':'/api/health','docs':'docs/API_REFERENCE.md'})
@app.errorhandler(413)
def too_large(_): return jsonify({'error':'Upload/request exceeds the 8 MB server limit.'}),413
@app.errorhandler(404)
def not_found(_): return jsonify({'error':'Route not found.'}),404
@app.errorhandler(500)
def internal(_): return jsonify({'error':'Internal server error.'}),500
with app.app_context(): init_db()
if __name__=='__main__':
 # Debug stays off to avoid interactive debugger exposure.
 app.run(host=os.getenv('VYNDRA_HOST','127.0.0.1'),port=int(os.getenv('VYNDRA_PORT','5001')),debug=False)
