#!/usr/bin/env python3
"""
Command Line Chat Application using Google Gemini
A simple interactive chat interface for conversing with Gemini AI.
"""

import os
import sys
import warnings
from typing import List, Optional
from datetime import datetime

warnings.filterwarnings("ignore", category=ResourceWarning)

try:
    from google import genai
except ImportError:
    print("Error: google-genai package not found. Please install it with: pip install google-genai")
    sys.exit(1)


class GeminiChat:
    """Interactive chat application using Google Gemini."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """Initialize the chat application."""
        self.api_key = api_key
        self.model_name = model_name
        self.conversation_history: List[dict] = []
        self.client = None
        
    def start_chat(self) -> None:
        """Start the interactive chat session."""
        print("🤖 Gemini Chat Application")
        print("=" * 50)
        print("Type your messages below. Commands:")
        print("  /help    - Show this help message")
        print("  /clear   - Clear conversation history")
        print("  /exit    - Exit the chat")
        print("  /history - Show conversation history")
        print("=" * 50)
        print()
        
        try:
            with genai.Client(api_key=self.api_key) as client:
                self.client = client
                self._chat_loop()
        except Exception as e:
            print(f"❌ Error connecting to Gemini: {e}")
            return
    
    def _chat_loop(self) -> None:
        """Main chat loop."""
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Handle empty input
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue
                
                # Add user message to history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Get response from Gemini
                response = self._get_gemini_response(user_input)
                
                if response:
                    print(f"\n🤖 Gemini: {response}")
                    print()
                    
                    # Add assistant response to history
                    self.conversation_history.append({
                        "role": "assistant", 
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    print("❌ Sorry, I couldn't get a response from Gemini.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                continue
    
    def _get_gemini_response(self, user_input: str) -> Optional[str]:
        """Get response from Gemini API."""
        try:
            # Build context from conversation history
            context = self._build_context()
            
            # Create the full prompt
            full_prompt = f"{context}\n\nUser: {user_input}"
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
            )
            
            return response.text if response else None
            
        except Exception as e:
            print(f"❌ Error getting response: {e}")
            return None
    
    def _build_context(self) -> str:
        """Build context from conversation history."""
        if not self.conversation_history:
            return "You are a helpful AI assistant. Please respond to the user's questions and requests in a friendly, informative manner."
        
        # Get last few exchanges for context (to avoid token limits)
        recent_history = self.conversation_history[-10:]  # Last 10 messages
        
        context_parts = ["You are a helpful AI assistant. Here's our recent conversation:"]
        
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def _handle_command(self, command: str) -> None:
        """Handle chat commands."""
        command = command.lower().strip()
        
        if command == '/help':
            self._show_help()
        elif command == '/clear':
            self._clear_history()
        elif command == '/exit':
            print("👋 Goodbye!")
            sys.exit(0)
        elif command == '/history':
            self._show_history()
        else:
            print(f"❌ Unknown command: {command}")
            print("Type /help for available commands.")
    
    def _show_help(self) -> None:
        """Show help information."""
        print("\n📋 Available Commands:")
        print("  /help    - Show this help message")
        print("  /clear   - Clear conversation history")
        print("  /exit    - Exit the chat")
        print("  /history - Show conversation history")
        print()
    
    def _clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        print("✅ Conversation history cleared.")
        print()
    
    def _show_history(self) -> None:
        """Show conversation history."""
        if not self.conversation_history:
            print("📝 No conversation history yet.")
            return
        
        print("\n📝 Conversation History:")
        print("-" * 50)
        
        for i, msg in enumerate(self.conversation_history, 1):
            role = "You" if msg["role"] == "user" else "Gemini"
            timestamp = msg["timestamp"][:19]  # Remove microseconds
            print(f"{i}. [{timestamp}] {role}: {msg['content']}")
        
        print("-" * 50)
        print()


def main() -> None:
    """Main function to run the chat application."""
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set.")
        print("Please set your API key:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        print("\nOr run: python chat.py --api-key YOUR_API_KEY")
        return
    
    # Create and start chat
    chat = GeminiChat(api_key)
    chat.start_chat()


if __name__ == "__main__":
    main()
