from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Agent Functions ===
def research_agent(topic):
    prompt = f"Find detailed information about: {topic}. Include statistics, examples, and challenges."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def summarizer_agent(research_text):
    prompt = f"Summarize the following text into clear bullet points:\n\n{research_text}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def writer_agent(summary_points):
    prompt = f"Write a 300-word blog post based on these bullet points:\n\n{summary_points}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# === Main Multi-Agent Coordinator ===
def multi_agent_system(topic):
    print("🔍 Research Agent: Gathering info...")
    research_data = research_agent(topic)

    print("✏ Summarizer Agent: Creating bullet points...")
    summary = summarizer_agent(research_data)

    print("📝 Writer Agent: Producing final content...")
    final_article = writer_agent(summary)

    return final_article

# === Run Example ===
if __name__ == "__main__":
    user_topic = input("Enter a topic: ")
    result = multi_agent_system(user_topic)
    print("\n=== Final Output ===\n")
    print(result)
