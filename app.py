from flask import Flask, request, jsonify
from flask_cors import CORS
from ARGO_backend import EnhancedARGOChatbot

app = Flask(__name__)
CORS(app)

# Initialize ARGO chatbot once when the backend starts
bot = EnhancedARGOChatbot()

@app.route('/api/data', methods=['GET'])
def get_data_summary():
    return jsonify({"summary": bot.get_data_summary()})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    response = bot.chat(question)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
