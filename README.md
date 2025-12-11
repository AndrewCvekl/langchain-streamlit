# 🎵 Music Store Customer Support Bot

An intelligent customer support chatbot for a music store, built with LangChain, LangGraph, and Streamlit. Features SMS verification, lyrics search, YouTube video embedding, and comprehensive music catalog browsing.

## ✨ Features

### 🎤 Lyrics Search (NEW!)
- Search for songs by remembering lyrics snippets
- Powered by Genius API
- Check if songs are in your catalogue
- Watch YouTube videos directly in the chat
- Get purchase recommendations or provide feedback

### 🔒 Secure Account Management
- SMS verification via Twilio for sensitive operations
- Update email addresses and mailing addresses securely
- View account details and purchase history

### 🎼 Music Catalog
- Browse 3,500+ tracks in the Chinook database
- Search by artist, album, genre, or track name
- View pricing and track details
- Check purchase history

### 🤖 AI-Powered Agent
- Built with LangGraph for robust agentic workflows
- Autonomous tool selection and decision making
- Natural conversation flow
- Context-aware responses

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- API Keys (see Setup)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AndrewCvekl/langchain-streamlit.git
cd langchain-streamlit
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### API Keys Required

1. **OpenAI API** - Get from [OpenAI Platform](https://platform.openai.com/)
2. **Genius API** - Get from [Genius API Clients](https://genius.com/api-clients)
3. **YouTube Data API v3** - Get from [Google Cloud Console](https://console.cloud.google.com/)
4. **Twilio** (for SMS) - Get from [Twilio Console](https://www.twilio.com/console)

### Run the App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## 🎯 Usage Examples

### Lyrics Search
```
User: "I heard a song that goes 'can't you see that I'm the one who understands you'"
Bot: [Searches Genius API and presents matches]
User: "Yes, the first one!"
Bot: [Checks catalogue, offers video preview, and purchase option]
```

### Account Management
```
User: "I want to change my email address"
Bot: "For security, I need to verify your identity. I'll send a code to your phone."
[SMS sent via Twilio]
User: "My code is 123456"
Bot: [Verifies and allows email update]
```

### Music Browsing
```
User: "Show me Rock albums"
Bot: [Lists Rock albums from catalogue with details]
```

## 🏗️ Architecture

### Tech Stack
- **Frontend**: Streamlit
- **AI Framework**: LangChain + LangGraph
- **LLM**: OpenAI GPT-4
- **Database**: SQLite (Chinook)
- **APIs**: Genius, YouTube Data API v3, Twilio

### Project Structure
```
├── app.py                      # Streamlit UI
├── graph_with_verification.py  # LangGraph workflow
├── tools_v2.py                 # Music & lyrics tools
├── tools_account.py            # Account management tools
├── verification.py             # SMS verification service
├── database.py                 # Database utilities
├── chinook.db                  # SQLite database (3,500+ tracks)
└── requirements.txt            # Python dependencies
```

### LangGraph Workflow
```
User Input → General Agent → Route Decision
                           ↓
            ┌──────────────┴──────────────┐
            ↓                             ↓
    Music Agent                    Customer Agent
    (Lyrics, Catalog)              (Account, Verification)
            ↓                             ↓
    Tool Execution ←──────────────────────┘
            ↓
    Response to User
```

## 🛠️ Development

### Adding New Tools

Create tools in `tools_v2.py`:

```python
@tool
def your_new_tool(param: str) -> str:
    """
    Clear description for the LLM.
    
    Args:
        param: Description of parameter
        
    Returns:
        Description of return value
    """
    # Your implementation
    return result
```

Add to `ALL_TOOLS` list in `tools_v2.py`.

### Modifying the Workflow

Edit the system prompt in `graph_with_verification.py` to guide the agent's behavior.

## 🔐 Security

- **SMS Verification**: Required for sensitive account changes
- **Environment Variables**: All secrets in `.env` (never committed)
- **Input Validation**: SQL injection protection via parameterized queries
- **API Rate Limiting**: Built-in retry logic and timeouts

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [Chinook Database](https://github.com/lerocha/chinook-database) - Sample music store database
- [LangChain](https://langchain.com/) - AI framework
- [Streamlit](https://streamlit.io/) - Web framework
- Genius API - Lyrics search
- YouTube Data API - Video embedding

## 📧 Contact

Andrew Cvekl - [@AndrewCvekl](https://github.com/AndrewCvekl)

Project Link: [https://github.com/AndrewCvekl/langchain-streamlit](https://github.com/AndrewCvekl/langchain-streamlit)
