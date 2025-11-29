# 🎓 Course TA Agent Studio

A multi-tenant platform for creating, hosting, and chatting with course-specific **Teaching Assistant AI agents**. Professors can create TA agents that connect to their **Google Drive folders** containing course materials (syllabi, slides, homework, etc.), automatically index the content using RAG (Retrieval-Augmented Generation), and provide students with an intelligent chat interface.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

### For Professors
- **Easy Agent Creation**: Create TA agents via a simple web UI with just a name, Google Drive folder, and API key
- **Multiple LLM Providers**: Choose between **OpenAI** (GPT-4o-mini, GPT-5) or **Google Gemini** (2.0-flash, 2.5-flash, 2.5-pro)
- **Customizable Persona**: Define how your TA should behave and respond
- **Announcements**: Post important announcements that appear to all students
- **Analytics Dashboard**: View query logs and understand what students are asking
- **Auto-Refresh**: Content automatically re-indexes daily to pick up new materials

### For Students
- **Shareable Chat Links**: Access via simple URLs like `/public/{agent-slug}`
- **Smart Suggestions**: AI-generated question suggestions based on course content
- **Voice Input**: Speak your questions using speech recognition
- **Streaming Responses**: Real-time token-by-token response streaming
- **Conversation Memory**: Context-aware responses with conversation summarization

### Technical Highlights
- **RAG Pipeline**: ChromaDB vector store with per-agent collections
- **Multi-format Support**: PDFs, Google Docs, Slides, Sheets, and images (OCR)
- **Embedding Isolation**: Each agent uses its own embedding model configuration
- **Upfront Validation**: API keys and Google Drive folders are validated before agent creation
- **Progress Tracking**: Real-time progress display during document indexing

---

## 📋 Prerequisites

- **Python 3.11+**
- **Google Cloud Project** with Drive API enabled
- **API Key** for OpenAI or Google Gemini
- **Tesseract OCR** (optional, for image text extraction)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/EECE798S_Project.git
cd EECE798S_Project
```

### 2. Create a Virtual Environment

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Required
SECRET_KEY=your-secret-key-here  # Generate with: python -c "import secrets; print(secrets.token_hex(32))"

# Google OAuth (required for Drive access)
GOOGLE_API_CLIENT_ID=your-google-client-id
GOOGLE_API_CLIENT_SECRET=your-google-client-secret

# Optional: Default API keys (users can also provide per-agent keys)
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# Optional: Scheduler configuration
SCHED_CRON=0 3 * * *  # Daily refresh at 3 AM

# Optional: Base URL for OAuth redirects
BASE_URL=http://127.0.0.1:8000
```

### 5. Set Up Google Cloud OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Google Drive API**
4. Go to **Credentials** → **Create Credentials** → **OAuth Client ID**
5. Choose **Desktop app** or **Web application**
6. Download the credentials JSON and save it as `data/google_client_secret.json`

### 6. Run the Application

```bash
uvicorn app.main:app --reload
```

Visit **http://127.0.0.1:8000** in your browser.

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Using Docker Directly

```bash
# Build the image
docker build -t course-ta-agent .

# Run the container
docker run -d \
  --name course-ta-agent \
  -p 8000:8000 \
  -v ./data:/app/data \
  --env-file .env \
  course-ta-agent
```

### Docker Environment Variables

The Docker container expects these environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `SECRET_KEY` | Session signing key | Yes |
| `GOOGLE_API_CLIENT_ID` | Google OAuth client ID | Yes |
| `GOOGLE_API_CLIENT_SECRET` | Google OAuth client secret | Yes |
| `OPENAI_API_KEY` | OpenAI API key | No |
| `GEMINI_API_KEY` | Gemini API key | No |
| `DATA_DIR` | Data directory path (default: `/app/data`) | No |
| `TESSERACT_CMD` | Tesseract binary path | No |

---

## 📁 Project Structure

```
EECE798S_Project/
├── app/                      # Main application code
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── agents.py            # Agent CRUD operations
│   ├── auth.py              # User authentication
│   ├── chat.py              # Chat endpoints & streaming
│   ├── db.py                # Database configuration
│   ├── models.py            # SQLAlchemy models
│   ├── providers.py         # OpenAI & Gemini providers
│   ├── rag.py               # RAG indexing & retrieval
│   ├── progress.py          # Embedding progress tracking
│   ├── scheduler.py         # Background refresh jobs
│   ├── security.py          # Session & password handling
│   └── settings.py          # Environment settings
├── templates/               # Jinja2 HTML templates
│   ├── home.html           # Landing page
│   ├── dashboard.html      # Professor dashboard
│   ├── chat.html           # Private chat interface
│   └── public_chat.html    # Public student chat
├── tests/                   # Unit tests
│   ├── conftest.py         # Pytest fixtures
│   ├── test_agents.py
│   ├── test_chat.py
│   ├── test_models.py
│   ├── test_progress.py
│   ├── test_providers.py
│   ├── test_rag.py
│   └── test_security.py
├── data/                    # Persistent data (gitignored)
│   ├── app.db              # SQLite database
│   ├── chroma/             # Vector store
│   └── google_tokens/      # OAuth tokens
├── eval/                    # RAG evaluation scripts
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 🧪 Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=app --cov-report=html
```

### Run Specific Test File

```bash
pytest tests/test_providers.py -v
```

---

## 📖 Usage Guide

### Creating an Agent (Professor)

1. **Sign Up / Log In** at the home page
2. **Connect Google Drive** using the OAuth flow
3. Click **"Create New Agent"**
4. Fill in the form:
   - **Name**: Display name for your course TA
   - **Google Drive Folder**: URL or ID of folder with course materials
   - **Provider**: Choose OpenAI or Gemini
   - **Model**: Select chat model (e.g., gpt-4o-mini, gemini-2.5-flash)
   - **Embedding Model**: Select embedding model for RAG
   - **API Key**: Your provider API key
   - **Persona** (optional): Custom behavior instructions
5. Click **Create** and wait for indexing to complete

### Sharing with Students

After creating an agent, you'll get a shareable link like:
```
http://your-domain.com/public/{agent-slug}
```

Share this link with your students. No login required!

### Updating Course Materials

1. Add new files to your Google Drive folder
2. The system auto-refreshes daily, or
3. Go to agent settings and click **"Re-index"**

---

## 🔧 Configuration Options

### Supported Models

| Provider | Chat Models | Embedding Models |
|----------|-------------|------------------|
| OpenAI | gpt-4.1-nano, gpt-4o-mini, gpt-5-mini, gpt-5-nano, gpt-5 | text-embedding-3-small, text-embedding-ada-002 |
| Gemini | gemini-2.0-flash-lite, gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro | models/text-embedding-004, models/gemini-embedding-001 |

### Scheduler Configuration

The `SCHED_CRON` environment variable accepts cron syntax:

```
SCHED_CRON=0 3 * * *    # Every day at 3:00 AM
SCHED_CRON=0 */6 * * *  # Every 6 hours
SCHED_CRON=0 0 * * 0    # Every Sunday at midnight
```

---

## 🛠️ Troubleshooting

### Common Issues

**"Google Drive not connected"**
- Ensure you've completed the OAuth flow
- Check that `data/google_client_secret.json` exists

**"API Key Invalid"**
- Verify your API key is correct and has credits
- OpenAI keys start with `sk-`
- Gemini keys start with `AI`

**"Folder is empty"**
- Ensure your Google Drive folder contains files
- Make sure the folder is shared with your connected account

**"OCR not working"**
- Install Tesseract: `apt-get install tesseract-ocr` (Linux) or download from [GitHub](https://github.com/tesseract-ocr/tesseract)
- Set `TESSERACT_CMD` environment variable

### Viewing Logs

```bash
# Local development
uvicorn app.main:app --reload --log-level debug

# Docker
docker-compose logs -f app
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `pytest`
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push to branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [LangChain](https://langchain.com/) - LLM application framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [OpenAI](https://openai.com/) & [Google Gemini](https://deepmind.google/technologies/gemini/) - LLM providers

---

## 📧 Support

For questions or issues, please open a GitHub issue or contact the course instructor.

---

*Built as a course project for EECE 798S at the American University of Beirut.*
