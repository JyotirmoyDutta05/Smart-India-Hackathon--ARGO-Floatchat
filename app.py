from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ARGO_backend import EnhancedARGOChatbot
import os
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ARGO chatbot once when the backend starts
try:
    bot = EnhancedARGOChatbot()
    logger.info("ARGO Chatbot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ARGO Chatbot: {e}")
    bot = None

@app.route('/api/data', methods=['GET'])
def get_data_summary():
    """Get dataset summary"""
    if bot is None:
        return jsonify({"error": "Chatbot not initialized"}), 500
    
    try:
        summary = bot.get_data_summary()
        return jsonify({"summary": summary})
    except Exception as e:
        logger.error(f"Error getting data summary: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests with support for both text and image responses"""
    if bot is None:
        return jsonify({"error": "Chatbot not initialized"}), 500
    
    try:
        data = request.json
        question = data.get("question", "")
        
        if not question:
            return jsonify({"error": "No question provided"}), 400

        logger.info(f"Processing question: {question[:100]}...")
        
        # Get response from chatbot
        response = bot.chat(question)
        
        # Check if response is a dictionary (image/plot response)
        if isinstance(response, dict):
            if response.get('type') == 'image':
                # Image response - return image data with metadata
                return jsonify({
                    "type": "image",
                    "filename": response.get('filename'),
                    "url": response.get('url'),
                    "base64": response.get('base64'),
                    "data_points": response.get('data_points'),
                    "description": response.get('description'),
                    "message": f"Generated {response.get('description', 'plot')} successfully!"
                })
            elif response.get('type') == 'error':
                # Error response from plotting
                return jsonify({
                    "type": "text", 
                    "content": response.get('message', 'An error occurred while generating the plot.')
                })
            else:
                # Other dictionary responses
                return jsonify(response)
        else:
            # Text response
            return jsonify({
                "type": "text", 
                "content": response
            })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            "type": "text", 
            "content": f"Sorry, I encountered an error while processing your request: {str(e)}"
        }), 500

@app.route('/static/<filename>')
def serve_static_file(filename):
    """Serve static files (images)"""
    try:
        return send_from_directory('static', filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {e}")
        return jsonify({"error": "File not found"}), 404

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "chatbot_initialized": bot is not None,
        "static_directory": os.path.exists('static')
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Run the app
    app.run(host="0.0.0.0", port=8000, debug=True)
