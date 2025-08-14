import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
import os

# === Load API Key from .env ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ No API key found in .env file")
client = OpenAI(api_key=OPENAI_API_KEY)

# === Load and Analyze Data ===
def analyze_file(file_path):
    # Detect file type
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    print(f"✅ File loaded: {file_path}")
    print(f"📊 Shape: {df.shape}")
    print(f"📝 Columns: {list(df.columns)}\n")

    # Basic stats
    summary = df.describe(include="all").to_string()

    # Save a visualization example
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) >= 1:
        df[numeric_cols[0]].hist()
        plt.title(f"Distribution of {numeric_cols[0]}")
        plt.savefig("histogram.png")
        print("📈 Histogram saved as histogram.png")

    return df, summary

# === AI Summary ===
def ai_summary(df_summary):
    prompt = f"""
You are a data analyst. Summarize the following dataset statistics in plain English.
Here are the details:
{df_summary}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# === Main Program ===
def main():
    file_path = input("📂 Enter path to CSV/Excel file: ").strip()
    df, summary = analyze_file(file_path)
    insight = ai_summary(summary)
    print("\n💡 Insights from AI:")
    print(insight)

if __name__ == "__main__":
    main()
