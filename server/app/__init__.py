from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()


def create_app():
    app = Flask(__name__)
    # Enable CORS with specific options
    CORS(app, resources={
         r"/*": {"origins": "*", "supports_credentials": True}})

    # Register routes
    from .routes import main
    app.register_blueprint(main)

    # Start the PCA update scheduler
    from .scheduler import start_scheduler
    start_scheduler()

    # Register shutdown handler
    @app.teardown_appcontext
    def shutdown_scheduler(exception=None):
        from .scheduler import stop_scheduler
        stop_scheduler()

    return app
