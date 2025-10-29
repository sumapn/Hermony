import os
import warnings
from typing import Final

warnings.filterwarnings("ignore", category=ResourceWarning)

from google import genai
# Define the prompt
user_prompt: Final[str] = "I have headaches and I'm not sure what to do about it."
model_name: Final[str] = "gemini-2.5-flash"

# System instruction to guide Gemini's behavior
system_prompt: Final[str] = (
    "You are an expert doctor specialised on menopause. The user is a female who is possibly "
    "going through menopause issues. You are concerned about her wellbeing however you are "
    "reluctant to give prescriptions since it can have legal consequences. Converse with her "
    "gently and give general advice only based on your understanding of her condition. Also chat "
    "like a friend and ask one question at a time. When the user appears to end the conversation, "
    "end the conversation with a reassuring tone. Also ensure that the conversation does not go "
    "rambling. Not more than 5 questions for the first time the symptom is being logged, and less "
    "than 3 questions for the 2nd time logging onwards unless the women want to chat more."
)

def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Please export it in your environment before running."
        )

    # 1. Use the 'with' statement for safe client management.
    # The 'with' block ensures the client connection is closed automatically.
    try:
        with genai.Client(api_key=api_key) as client:
            print(f"Sending prompt to {model_name}: '{user_prompt}'\n")

            # 2. Generate the content inside the 'with' block
            response = client.models.generate_content(
                model=model_name,
                contents=f"{system_prompt}\n\nUser: {user_prompt}",
            )

            # 3. Print the model's response text
            print("Gemini's Response:")
            print("-" * 20)
            print(response.text)

    # Handle potential errors
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()