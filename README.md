# 📄 RAG Document Chatbot

An AI-powered document Q&A chatbot that lets users upload any PDF and ask questions in natural language. Built with LangChain, ChromaDB, and Google Gemini — deployed live on the web.

---

## 🚀 Live Demo
🔗 [Coming Soon — Deploying on Streamlit Cloud]

---

## 💡 What It Does
- Upload any PDF document
- Ask questions in natural language
- Get accurate answers pulled directly from the document
- Powered by Retrieval-Augmented Generation (RAG) pipeline

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM | Google Gemini (langchain-google-genai) |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| PDF Processing | PyPDF |
| Environment | python-dotenv |
| Language | Python 3 |

---

## 🧠 How It Works

```
User uploads PDF
      ↓
PDF is split into chunks
      ↓
Chunks are embedded and stored in ChromaDB (Vector Store)
      ↓
User asks a question
      ↓
Relevant chunks are retrieved from ChromaDB
      ↓
Google Gemini generates answer based on retrieved chunks
      ↓
Answer displayed to user
```

---

## ⚙️ How To Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/kumardatascience/rag-document-chatbot.git
cd rag-document-chatbot
```

**2. Create a virtual environment**
```bash
python -m venv rag-env
source rag-env/bin/activate  # On Windows: rag-env\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your API key**

Create a `.env` file in the root folder:
```
GOOGLE_API_KEY=your_google_api_key_here
```

**5. Run the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
rag-document-chatbot/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not pushed to GitHub)
├── .gitignore           # Ignores .env and other sensitive files
└── README.md            # Project documentation
```

---

## 🔐 Security
- API keys are stored in `.env` file
- `.env` is added to `.gitignore` — never pushed to GitHub

---

## 👤 Author
**Kumar Katariya**
- Upwork: [kumardatascience](https://www.upwork.com)
- GitHub: [kumardatascience](https://github.com/kumardatascience)
- Kaggle: Top 4% Globally

---

## 📌 Related Projects
- [Chest X-Ray Pneumonia Detector](https://github.com/kumardatascience) — ResNet18 Transfer Learning, 86.22% accuracy
- [Bankruptcy Prediction System](https://github.com/kumardatascience) — XGBoost ROC-AUC 0.9367
