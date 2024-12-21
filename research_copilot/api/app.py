from flask import Flask
from flask_cors import CORS

def create_app(config_object=None):
    app = Flask(__name__)
    CORS(app)
    
    if config_object:
        app.config.from_object(config_object)
    
    # Register blueprints here
    
    return app