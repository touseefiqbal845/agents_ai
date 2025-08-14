from __future__ import annotations
import os
import openai
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from PIL import Image
import pytesseract
import fitz  # PyMuPDF for PDFs
import speech_recognition as sr
import requests
from bs4 import BeautifulSoup
import json
import datetime
import schedule
import time
import threading
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import hashlib
import sqlite3
from typing import List, Dict, Any, Optional
import logging

# ====== CONFIG ======
OPENAI_API_KEY = "Kt9cpecoNyy3YtdAVgV5UrHCxeBc6SDF5mxrMGfz40QjLvpw6UKo9E9WMnGY6hQA"
CHROMA_API_KEY = "ck-CbZnvFgNiZrbkDZt2JA4"
CHROMA_TENANT = "96b80992-35166049b90112f"
CHROMA_DB_NAME = "tousvector_db"

openai.api_key = OPENAI_API_KEY

# ====== LOGGING SETUP ======
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_activity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ====== INIT CHROMA WITH OPENAI EMBEDDINGS ======
openai_ef = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

vector_client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DB_NAME
)

collection = vector_client.get_or_create_collection(
    "long_term_memory_new",
    embedding_function=openai_ef
)

# ====== SQLITE DATABASE FOR TASKS AND ANALYTICS ======
def init_database():
    conn = sqlite3.connect('agent_data.db')
    cursor = conn.cursor()
    
    # Tasks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'pending',
            priority TEXT DEFAULT 'medium',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            due_date TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    
    # Analytics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_type TEXT,
            content_hash TEXT,
            sentiment_score REAL,
            processing_time REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# ====== ENHANCED MEMORY FUNCTIONS ======
def store_memory(text, metadata=None):
    content_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Store in ChromaDB
    collection.add(
        documents=[text],
        ids=[content_hash],
        metadatas=[metadata or {}]
    )
    
    # Store analytics
    if metadata:
        conn = sqlite3.connect('agent_data.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO analytics (interaction_type, content_hash, processing_time)
            VALUES (?, ?, ?)
        ''', (metadata.get('type', 'text'), content_hash, metadata.get('processing_time', 0)))
        conn.commit()
        conn.close()

def get_relevant_memories(query, top_k=3):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results["documents"][0] if results["documents"] else []

def search_memories_by_date(start_date, end_date):
    """Search memories within a date range"""
    # This would require storing timestamps in metadata
    # For now, return recent memories
    results = collection.get()
    return results["documents"][-10:] if results["documents"] else []

# ====== ENHANCED MULTI-MODAL INPUT PROCESSORS ======
def process_image(path):
    start_time = time.time()
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        
        # Extract image metadata
        metadata = {
            'type': 'image',
            'size': img.size,
            'mode': img.mode,
            'processing_time': time.time() - start_time
        }
        
        result = f"Extracted from image: {text.strip()}"
        store_memory(result, metadata)
        return result
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return f"Error processing image: {str(e)}"

def process_pdf(path):
    start_time = time.time()
    try:
        pdf = fitz.open(path)
        text = ""
        page_count = len(pdf)
        
        for page_num, page in enumerate(pdf):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.get_text()
        
        metadata = {
            'type': 'pdf',
            'page_count': page_count,
            'processing_time': time.time() - start_time
        }
        
        result = f"Extracted from PDF ({page_count} pages): {text.strip()}"
        store_memory(result, metadata)
        return result
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return f"Error processing PDF: {str(e)}"

def process_voice(path):
    start_time = time.time()
    try:
        r = sr.Recognizer()
        with sr.AudioFile(path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        
        metadata = {
            'type': 'voice',
            'processing_time': time.time() - start_time
        }
        
        result = f"Transcribed from audio: {text.strip()}"
        store_memory(result, metadata)
        return result
    except Exception as e:
        logger.error(f"Error processing voice: {e}")
        return f"Error processing voice: {str(e)}"

# ====== NEW FUNCTIONALITIES ======

def web_scrape(url):
    """Scrape content from a webpage"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        metadata = {
            'type': 'web_scrape',
            'url': url,
            'title': soup.title.string if soup.title else 'No title'
        }
        
        result = f"Scraped from {url}: {text[:1000]}..."
        store_memory(result, metadata)
        return result
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return f"Error scraping {url}: {str(e)}"

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    
    if sentiment_score > 0.1:
        sentiment = "positive"
    elif sentiment_score < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        'sentiment': sentiment,
        'score': sentiment_score,
        'subjectivity': blob.sentiment.subjectivity
    }

def generate_code(prompt, language="python"):
    """Generate code based on description"""
    messages = [
        {"role": "system", "content": f"You are a {language} programming expert. Generate clean, well-commented code."},
        {"role": "user", "content": prompt}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1000
    )
    
    return response.choices[0].message["content"]

def analyze_data(file_path):
    """Analyze CSV/Excel data files"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return "Unsupported file format. Please provide CSV or Excel file."
        
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else None
        }
        
        return f"Data Analysis Results:\n{json.dumps(analysis, indent=2, default=str)}"
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        return f"Error analyzing data: {str(e)}"

def create_task(title, description, priority="medium", due_date=None):
    """Create a new task"""
    conn = sqlite3.connect('agent_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO tasks (title, description, priority, due_date)
        VALUES (?, ?, ?, ?)
    ''', (title, description, priority, due_date))
    
    task_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return f"Task created with ID: {task_id}"

def list_tasks(status="all"):
    """List tasks with optional status filter"""
    conn = sqlite3.connect('agent_data.db')
    cursor = conn.cursor()
    
    if status == "all":
        cursor.execute('SELECT * FROM tasks ORDER BY created_at DESC')
    else:
        cursor.execute('SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC', (status,))
    
    tasks = cursor.fetchall()
    conn.close()
    
    if not tasks:
        return "No tasks found."
    
    result = "Tasks:\n"
    for task in tasks:
        result += f"ID: {task[0]} | {task[1]} | Status: {task[3]} | Priority: {task[4]}\n"
    
    return result

def complete_task(task_id):
    """Mark a task as completed"""
    conn = sqlite3.connect('agent_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE tasks SET status = 'completed', completed_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (task_id,))
    
    conn.commit()
    conn.close()
    
    return f"Task {task_id} marked as completed"

def schedule_reminder(message, delay_minutes):
    """Schedule a reminder"""
    def reminder():
        print(f"⏰ REMINDER: {message}")
    
    schedule.every(delay_minutes).minutes.do(reminder)
    return f"Reminder scheduled for {delay_minutes} minutes from now"

def get_weather(city):
    """Get weather information (placeholder - would need API key)"""
    # This is a placeholder - you'd need to integrate with a weather API
    return f"Weather information for {city} would be displayed here. (API integration required)"

def translate_text(text, target_language="es"):
    """Translate text to target language"""
    messages = [
        {"role": "system", "content": f"You are a translator. Translate the following text to {target_language}."},
        {"role": "user", "content": text}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    return response.choices[0].message["content"]

def summarize_text(text, max_length=200):
    """Summarize long text"""
    messages = [
        {"role": "system", "content": f"Summarize the following text in {max_length} characters or less:"},
        {"role": "user", "content": text}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    return response.choices[0].message["content"]

# ====== ADVANCED FEATURES ======

def generate_image(prompt, size="1024x1024"):
    """Generate image using DALL-E"""
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size=size
        )
        image_url = response['data'][0]['url']
        return f"Generated image URL: {image_url}"
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return f"Error generating image: {str(e)}"

def file_operations(operation, file_path, content=None):
    """Perform file operations"""
    try:
        if operation == "read":
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif operation == "write":
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"File written to {file_path}"
        elif operation == "append":
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)
            return f"Content appended to {file_path}"
        elif operation == "delete":
            os.remove(file_path)
            return f"File {file_path} deleted"
        else:
            return "Invalid operation. Use: read, write, append, delete"
    except Exception as e:
        logger.error(f"File operation error: {e}")
        return f"File operation error: {str(e)}"

def get_system_info():
    """Get system information"""
    import platform
    import psutil
    
    info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'memory_usage': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'cpu_usage': psutil.cpu_percent()
    }
    
    return f"System Information:\n{json.dumps(info, indent=2)}"

def create_backup():
    """Create backup of agent data"""
    import shutil
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_{timestamp}"
    
    try:
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup database
        if os.path.exists('agent_data.db'):
            shutil.copy2('agent_data.db', f"{backup_dir}/agent_data.db")
        
        # Backup log file
        if os.path.exists('agent_activity.log'):
            shutil.copy2('agent_activity.log', f"{backup_dir}/agent_activity.log")
        
        return f"Backup created in {backup_dir}"
    except Exception as e:
        logger.error(f"Backup error: {e}")
        return f"Backup error: {str(e)}"

def get_analytics():
    """Get analytics about agent usage"""
    conn = sqlite3.connect('agent_data.db')
    cursor = conn.cursor()
    
    # Get interaction counts by type
    cursor.execute('''
        SELECT interaction_type, COUNT(*) as count, 
               AVG(sentiment_score) as avg_sentiment,
               AVG(processing_time) as avg_time
        FROM analytics 
        GROUP BY interaction_type
    ''')
    
    analytics = cursor.fetchall()
    conn.close()
    
    if not analytics:
        return "No analytics data available."
    
    result = "Analytics Summary:\n"
    for row in analytics:
        result += f"Type: {row[0]} | Count: {row[1]} | Avg Sentiment: {row[2]:.2f} | Avg Time: {row[3]:.2f}s\n"
    
    return result

def search_files(directory, pattern):
    """Search for files matching pattern"""
    import glob
    import os
    
    try:
        search_path = os.path.join(directory, pattern)
        files = glob.glob(search_path, recursive=True)
        
        if not files:
            return f"No files found matching pattern: {pattern}"
        
        result = f"Found {len(files)} files:\n"
        for file in files[:10]:  # Limit to first 10 results
            result += f"- {file}\n"
        
        if len(files) > 10:
            result += f"... and {len(files) - 10} more files"
        
        return result
    except Exception as e:
        logger.error(f"File search error: {e}")
        return f"File search error: {str(e)}"

def encrypt_text(text, key="default_key"):
    """Simple text encryption"""
    import base64
    
    try:
        encoded = base64.b64encode(text.encode()).decode()
        return f"Encrypted: {encoded}"
    except Exception as e:
        return f"Encryption error: {str(e)}"

def decrypt_text(encrypted_text, key="default_key"):
    """Simple text decryption"""
    import base64
    
    try:
        if encrypted_text.startswith("Encrypted: "):
            encrypted_text = encrypted_text[11:]
        decoded = base64.b64decode(encrypted_text.encode()).decode()
        return f"Decrypted: {decoded}"
    except Exception as e:
        return f"Decryption error: {str(e)}"

# ====== ENHANCED CHAT FUNCTION ======
def chat_with_memory(user_input, context_type="general"):
    start_time = time.time()
    
    # Analyze sentiment
    sentiment = analyze_sentiment(user_input)
    
    # Get relevant memories
    relevant_memories = get_relevant_memories(user_input)
    context = "\n".join(relevant_memories) if relevant_memories else "No past memories."
    
    # Enhanced system prompt based on context
    system_prompts = {
        "general": "You are a multi-modal assistant with long-term memory, capable of processing text, images, PDFs, voice, web scraping, data analysis, and task management.",
        "coding": "You are a programming expert. Provide clear, well-commented code solutions.",
        "analysis": "You are a data analysis expert. Provide insights and recommendations based on data.",
        "task": "You are a task management assistant. Help organize and prioritize tasks effectively."
    }
    
    system_prompt = system_prompts.get(context_type, system_prompts["general"])
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nUser Query: {user_input}\n\nSentiment: {sentiment['sentiment']} (score: {sentiment['score']:.2f})"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    reply = response.choices[0].message["content"]
    
    # Store interaction with metadata
    metadata = {
        'type': context_type,
        'sentiment_score': sentiment['score'],
        'processing_time': time.time() - start_time,
        'context_type': context_type
    }
    
    store_memory(user_input + " -> " + reply, metadata)
    return reply

# ====== COMMAND PROCESSOR ======
def process_command(command, args):
    """Process special commands"""
    commands = {
        "help": lambda: print("""
Available Commands:
=== Basic Commands ===
- help: Show this help
- exit: Quit the program

=== Task Management ===
- tasks: List all tasks
- tasks pending: List pending tasks
- create task <title> | <description>: Create new task
- complete <task_id>: Mark task as complete
- reminder <minutes> <message>: Set reminder

=== Content Processing ===
- scrape <url>: Scrape webpage
- analyze <file_path>: Analyze data file
- code <language> <description>: Generate code
- translate <text> to <language>: Translate text
- summarize <text>: Summarize text
- sentiment <text>: Analyze sentiment
- memories: Show recent memories

=== File Operations ===
- file <operation> <file_path> [content]: File operations (read/write/append/delete)
- search <directory> <pattern>: Search for files
- backup: Create backup of agent data

=== AI & Generation ===
- generate <image_description>: Generate image using DALL-E
- encrypt <text>: Encrypt text
- decrypt <encrypted_text>: Decrypt text

=== System & Analytics ===
- system: Get system information
- analytics: Get usage analytics
- weather <city>: Get weather (placeholder)

=== Input Formats ===
- img:path - Process image
- pdf:path - Process PDF
- voice:path - Process audio
- scrape:url - Scrape webpage
- analyze:file - Analyze data file
- code:language:description - Generate code
- translate:text:to:language - Translate text
- create:task:title:description - Create task
        """),
        "tasks": lambda: list_tasks(args[0] if args else "all"),
        "create": lambda: create_task(args[0], args[1]) if len(args) >= 2 else "Usage: create task <title> | <description>",
        "complete": lambda: complete_task(int(args[0])) if args else "Usage: complete <task_id>",
        "reminder": lambda: schedule_reminder(" ".join(args[1:]), int(args[0])) if len(args) >= 2 else "Usage: reminder <minutes> <message>",
        "scrape": lambda: web_scrape(args[0]) if args else "Usage: scrape <url>",
        "analyze": lambda: analyze_data(args[0]) if args else "Usage: analyze <file_path>",
        "code": lambda: generate_code(" ".join(args[1:]), args[0]) if len(args) >= 2 else "Usage: code <language> <description>",
        "translate": lambda: translate_text(" ".join(args[:-2]), args[-1]) if len(args) >= 3 and args[-2] == "to" else "Usage: translate <text> to <language>",
        "summarize": lambda: summarize_text(" ".join(args)) if args else "Usage: summarize <text>",
        "sentiment": lambda: str(analyze_sentiment(" ".join(args))) if args else "Usage: sentiment <text>",
        "memories": lambda: str(search_memories_by_date(None, None)),
        "weather": lambda: get_weather(args[0]) if args else "Usage: weather <city>",
        "generate": lambda: generate_image(" ".join(args)) if args else "Usage: generate <image_description>",
        "file": lambda: file_operations(args[0], args[1], " ".join(args[2:]) if len(args) > 2 else None) if len(args) >= 2 else "Usage: file <operation> <file_path> [content]",
        "system": lambda: get_system_info(),
        "backup": lambda: create_backup(),
        "analytics": lambda: get_analytics(),
        "search": lambda: search_files(args[0], args[1]) if len(args) >= 2 else "Usage: search <directory> <pattern>",
        "encrypt": lambda: encrypt_text(" ".join(args)) if args else "Usage: encrypt <text>",
        "decrypt": lambda: decrypt_text(" ".join(args)) if args else "Usage: decrypt <encrypted_text>"
    }
    
    if command in commands:
        return commands[command]()
    else:
        return f"Unknown command: {command}. Type 'help' for available commands."

# ====== MAIN LOOP ======
def main():
    print("🤖 Enhanced Multi-Modal Long-Term Memory Agent Ready!")
    print("💡 Commands: 'help', 'img:path', 'pdf:path', 'voice:path', 'scrape:url', 'analyze:file'")
    print("💡 Special: 'code:language:description', 'translate:text:to:language', 'create:task:title:description'")
    print("💡 Type 'exit' to quit.")
    
    # Initialize database
    init_database()
    
    # Start scheduler in background
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == "exit":
                break
            
            # Check for commands
            if user_input.startswith("/"):
                command_parts = user_input[1:].split()
                if command_parts:
                    result = process_command(command_parts[0], command_parts[1:])
                    print("🤖:", result)
                continue
            
            # Process different input types
            if user_input.startswith("img:"):
                processed_input = process_image(user_input[4:].strip())
            elif user_input.startswith("pdf:"):
                processed_input = process_pdf(user_input[4:].strip())
            elif user_input.startswith("voice:"):
                processed_input = process_voice(user_input[6:].strip())
            elif user_input.startswith("scrape:"):
                processed_input = web_scrape(user_input[7:].strip())
            elif user_input.startswith("analyze:"):
                processed_input = analyze_data(user_input[8:].strip())
            elif user_input.startswith("code:"):
                parts = user_input[5:].split(":", 1)
                if len(parts) == 2:
                    processed_input = generate_code(parts[1], parts[0])
                else:
                    processed_input = "Usage: code:language:description"
            elif user_input.startswith("translate:"):
                parts = user_input[10:].split(":to:")
                if len(parts) == 2:
                    processed_input = translate_text(parts[0], parts[1])
                else:
                    processed_input = "Usage: translate:text:to:language"
            elif user_input.startswith("create:task:"):
                parts = user_input[12:].split(":", 1)
                if len(parts) == 2:
                    processed_input = create_task(parts[0], parts[1])
                else:
                    processed_input = "Usage: create:task:title:description"
            elif user_input.startswith("generate:"):
                processed_input = generate_image(user_input[9:].strip())
            elif user_input.startswith("encrypt:"):
                processed_input = encrypt_text(user_input[8:].strip())
            elif user_input.startswith("decrypt:"):
                processed_input = decrypt_text(user_input[8:].strip())
            else:
                processed_input = user_input
            
            reply = chat_with_memory(processed_input)
            print("🤖:", reply)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            print(f"🤖: An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
