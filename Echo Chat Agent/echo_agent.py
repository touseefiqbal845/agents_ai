from __future__ import annotations
import json
import os
from openai import OpenAI

load_dotenv()
# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


MEMORY_FILE = "memory.json"     # Conversation memory
OUTPUT_FILE = "output.json"     # Full API JSON metadata


# ==== MEMORY FUNCTIONS ====
def load_memory():
    """Load conversation history from file."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(memory):
    """Save conversation history to file."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def save_output_json(response_dict):
    """Save the full API response to a JSON file."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(response_dict, f, ensure_ascii=False, indent=2)

# ==== AI RESPONSE ====
def get_ai_response(memory, user_input):
    """Send message to OpenAI and return bot reply."""
    # Add user message to memory
    memory.append({"role": "user", "content": user_input})

    # Request from OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Or gpt-4o for higher quality
        messages=memory
    )

    # Convert API response to dict
    response_dict = response.model_dump()

    # Save full JSON output
    save_output_json(response_dict)

    # Extract assistant's reply
    bot_reply = response.choices[0].message.content

    # Add reply to memory
    memory.append({"role": "assistant", "content": bot_reply})

    return bot_reply

# ==== MAIN LOOP ====
def main():
    memory = load_memory()

    print("🤖 Echo Chat Agent — type 'exit' to quit")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        reply = get_ai_response(memory, user_input)
        print("AI:", reply)

        save_memory(memory)

if __name__ == "__main__":
    main()