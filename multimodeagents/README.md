# 🤖 Enhanced Multi-Modal Long-Term Memory Agent

A powerful AI assistant with multi-modal input processing, long-term memory, task management, and advanced analytics capabilities.

## 🚀 Features

### Core Capabilities
- **Multi-Modal Input Processing**: Text, images, PDFs, voice, web scraping
- **Long-Term Memory**: Persistent memory using ChromaDB vector database
- **Natural Language Processing**: Sentiment analysis, translation, summarization
- **Task Management**: Create, track, and manage tasks with reminders
- **Code Generation**: Generate code in multiple programming languages
- **Data Analysis**: Analyze CSV/Excel files with statistical insights

### Advanced Features
- **Image Generation**: Create images using DALL-E
- **File Operations**: Read, write, append, delete files
- **System Monitoring**: Get system information and resource usage
- **Backup & Recovery**: Automated backup of agent data
- **Analytics**: Usage analytics and performance metrics
- **File Search**: Search for files using patterns
- **Text Encryption/Decryption**: Basic text security features

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd multimodeagents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR (for image processing):
   - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

4. Set up your API keys in the configuration section of `multi_mode_agents.py`

## 🎯 Usage

### Basic Commands
```bash
python multi_mode_agents.py
```

### Input Formats

#### Text Processing
- Regular text input: `Hello, how are you?`
- Translation: `translate:Hello world:to:es`
- Summarization: `summarize:Long text content here`
- Sentiment analysis: `sentiment:This is great!`

#### File Processing
- Images: `img:path/to/image.png`
- PDFs: `pdf:path/to/document.pdf`
- Audio: `voice:path/to/audio.wav`
- Data files: `analyze:path/to/data.csv`

#### Web & Content
- Web scraping: `scrape:https://example.com`
- Code generation: `code:python:create a web scraper`
- Image generation: `generate:a beautiful sunset over mountains`

#### Task Management
- Create task: `create:task:Meeting:Prepare presentation for tomorrow`
- List tasks: `/tasks`
- Complete task: `/complete 1`

### Command Line Interface

Use `/` prefix for special commands:

#### Task Management
- `/tasks` - List all tasks
- `/tasks pending` - List pending tasks
- `/create task <title> | <description>` - Create new task
- `/complete <task_id>` - Mark task as complete
- `/reminder <minutes> <message>` - Set reminder

#### Content Processing
- `/scrape <url>` - Scrape webpage
- `/analyze <file_path>` - Analyze data file
- `/code <language> <description>` - Generate code
- `/translate <text> to <language>` - Translate text
- `/summarize <text>` - Summarize text
- `/sentiment <text>` - Analyze sentiment

#### File Operations
- `/file <operation> <file_path> [content]` - File operations
- `/search <directory> <pattern>` - Search for files
- `/backup` - Create backup of agent data

#### AI & Generation
- `/generate <image_description>` - Generate image using DALL-E
- `/encrypt <text>` - Encrypt text
- `/decrypt <encrypted_text>` - Decrypt text

#### System & Analytics
- `/system` - Get system information
- `/analytics` - Get usage analytics
- `/weather <city>` - Get weather (placeholder)

## 🔧 Configuration

### API Keys
Update the following variables in `multi_mode_agents.py`:

```python
OPENAI_API_KEY = "your_openai_api_key"
CHROMA_API_KEY = "your_chroma_api_key"
CHROMA_TENANT = "your_chroma_tenant"
CHROMA_DB_NAME = "your_chroma_db_name"
```

### Database
The system automatically creates:
- `agent_data.db` - SQLite database for tasks and analytics
- `agent_activity.log` - Activity log file

## 📊 Analytics

The system tracks:
- Interaction types and counts
- Sentiment scores
- Processing times
- System resource usage

View analytics with: `/analytics`

## 🔒 Security Features

- Text encryption/decryption
- Secure API key handling
- Activity logging
- Backup and recovery

## 🛠️ Advanced Usage

### Custom Extensions
You can extend the system by adding new functions and registering them in the command processor.

### Integration Examples
```python
# Process multiple files
for file in files:
    result = process_pdf(file)
    store_memory(result)

# Batch sentiment analysis
texts = ["Text 1", "Text 2", "Text 3"]
sentiments = [analyze_sentiment(text) for text in texts]
```

## 📝 Examples

### Example Session
```
🤖 Enhanced Multi-Modal Long-Term Memory Agent Ready!
💡 Commands: 'help', 'img:path', 'pdf:path', 'voice:path', 'scrape:url', 'analyze:file'
💡 Special: 'code:language:description', 'translate:text:to:language', 'create:task:title:description'
💡 Type 'exit' to quit.

You: /help
🤖: [Shows help menu]

You: scrape:https://example.com
🤖: Scraped from https://example.com: [content summary]...

You: code:python:create a simple calculator
🤖: [Generated Python code]

You: create:task:Review code:Check the generated calculator code
🤖: Task created with ID: 1

You: /analytics
🤖: Analytics Summary:
Type: web_scrape | Count: 1 | Avg Sentiment: 0.0 | Avg Time: 2.3s
Type: code | Count: 1 | Avg Sentiment: 0.2 | Avg Time: 1.8s
```

## 🐛 Troubleshooting

### Common Issues

1. **Tesseract not found**: Install Tesseract OCR
2. **API key errors**: Check your OpenAI API key
3. **Database errors**: Check file permissions
4. **Memory issues**: Monitor system resources

### Logs
Check `agent_activity.log` for detailed error information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenAI for GPT and DALL-E APIs
- ChromaDB for vector database
- All open-source libraries used in this project
