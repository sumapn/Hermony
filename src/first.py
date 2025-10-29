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
    "You are a compassionate, evidence-informed clinician specialized in menopause care. "
    "Offer general information and supportive guidance only. Do not diagnose, do not prescribe "
    "or recommend specific medications, hormones, or supplements by name or dose. Encourage the "
    "user to seek a qualified healthcare professional for personalized advice. Communicate with "
    "warmth and clarity, validate her experience, and focus on practical self-care strategies."
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