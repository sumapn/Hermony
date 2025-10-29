"""
Flask Web Server for Gemini Chat Application
A web-based chat interface for conversing with Gemini AI.
"""

import os
from typing import Any

from flask import Flask, render_template, request, jsonify, session, Response
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)

try:
    from google import genai
except ImportError:
    print("Error: google-genai package not found. Please install it with: pip install google-genai")
    exit(1)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')

# Initialize Gemini client
def get_gemini_client() -> genai.Client:
    """Get Gemini client instance."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)

def get_conversation_history() -> list[dict[str, Any]]:
    """Get conversation history from session."""
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    return session['conversation_history'] # type: ignore[reportOptionalMemberAccess]

def add_to_history(role: str, content: str) -> None:
    """Add message to conversation history."""
    history = get_conversation_history()
    history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    session['conversation_history'] = history

def build_context() -> str:
    """Build context from conversation history."""
    history = get_conversation_history()
    if not history:
        return "You are a helpful AI assistant. Please respond to the user's questions and requests in a friendly, informative manner."
    
    # Get last 10 messages for context
    recent_history = history[-10:]
    
    context_parts = ["You are a helpful AI assistant. Here's our recent conversation:"]
    
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        context_parts.append(f"{role}: {msg['content']}")
    
    return "\n".join(context_parts)

@app.route('/')
def index() -> Response:
    """Main chat page."""
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat() -> jsonify:
    """Handle chat messages."""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Add user message to history
        add_to_history('user', user_message)
        
        # Get response from Gemini
        with get_gemini_client() as client:
            context = build_context()
            full_prompt = f"{context}\n\nUser: {user_message}"
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
            )
            
            if response and response.text:
                gemini_response = response.text
                add_to_history('assistant', gemini_response)
                
                return jsonify({
                    'response': gemini_response,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({'error': 'Failed to get response from Gemini'}), 500
                
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/clear', methods=['POST'])
def clear_history() -> jsonify:
    """Clear conversation history."""
    session['conversation_history'] = []
    return jsonify({'message': 'History cleared successfully'})

@app.route('/history', methods=['GET'])
def get_history() -> jsonify:
    """Get conversation history."""
    history = get_conversation_history()
    return jsonify({'history': history})

@app.route('/health')
def health() -> jsonify:
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set.")
        print("Please set your API key:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        exit(1)
    
    print("🚀 Starting Flask server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=10000)

