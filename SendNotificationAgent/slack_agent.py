import os
from dotenv import load_dotenv
import requests

# === Load API Keys from .env ===
load_dotenv()
SLACK_TOKEN = os.getenv("SLACK_TOKEN")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL")

if not SLACK_TOKEN or not SLACK_CHANNEL:
    raise ValueError("❌ Missing SLACK_TOKEN or SLACK_CHANNEL in .env")

# === Send message to Slack ===
def send_slack_message(message):
    url = "https://slack.com/api/chat.postMessage"
    headers = {
        "Authorization": f"Bearer {SLACK_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "channel": SLACK_CHANNEL,
        "text": message
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200 and response.json().get("ok"):
        print(f"✅ Message sent: {message}")
    else:
        print(f"❌ Failed to send message: {response.text}")

# === Main Automation Logic ===
def automation_agent(command):
    """
    Simple rule-based agent:
    - Detects if user wants to send a Slack reminder
    - Adds safety confirmation before sending
    """
    command_lower = command.lower()

    if "reminder" in command_lower and "slack" in command_lower:
        print("📌 Detected: Slack reminder request.")
        confirm = input(f"⚠ Are you sure you want to send this? (yes/no): ").strip().lower()
        if confirm == "yes":
            send_slack_message(command)
        else:
            print("🚫 Action canceled.")
    else:
        print("🤖 I don't know how to handle that yet.")

# === Run Agent ===
if __name__ == "__main__":
    user_command = input("💬 What should I do? ").strip()
    automation_agent(user_command)
