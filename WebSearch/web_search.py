import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

# ====== LOAD ENV VARIABLES ======
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
 

if not OPENAI_API_KEY:
    raise ValueError("⚠️ OPENAI_API_KEY is missing. Add it to your .env file.")

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ====== SEARCH TOOL ======
def web_search(query):
    """Search the web using DuckDuckGo's free API."""
    params = {
        "q": query,
        "format": "json",
        "no_redirect": 1,
        "no_html": 1
    }
    response = requests.get("https://api.duckduckgo.com/", params=params)
    response.raise_for_status()
    data = response.json()

    search_data = []

    # Direct answer if available
    if data.get("AbstractText"):
        search_data.append(data["AbstractText"])

    # Related topics
    for topic in data.get("RelatedTopics", []):
        if isinstance(topic, dict) and "Text" in topic:
            search_data.append(topic["Text"])

    return "\n".join(search_data) if search_data else "No results found."

# ====== AI RESPONSE ======
def ai_answer_with_search(question):
    """Combine web search results with AI reasoning."""
    print("🔍 Searching the web...")
    search_results = web_search(question)

    prompt = f"""
    The user asked: {question}
    I searched the web and found these results:

    {search_results}

    Summarize the answer clearly and concisely using only these results.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a factual AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ====== MAIN LOOP ======
def main():
    print("🌐 Web Search Agent — type 'exit' to quit")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        answer = ai_answer_with_search(query)
        print("AI:", answer)

if __name__ == "__main__":
    main()
