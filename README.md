# Max-AI 🤖

An AI-powered chatbot built with Flask and integrated with Meta’s LLaMA 3.3 70B model.  
Max-AI can generate code, answer questions, and provide interactive responses through a modern web interface.

---

## 🚀 Features
- AI chatbot powered by LLaMA 3.3 70B.
- Code generation and interactive Q&A.
- Flask backend with secure session handling.
- Google OAuth login integration.
- Responsive frontend with personalized dashboard.
- Dragon animation for a creative user experience.

---

## 🛠️ Tech Stack
- **Backend:** Flask, Python
- **Frontend:** HTML, CSS, Jinja2 templates
- **AI Model:** Meta LLaMA 3.3 70B (via API)
- **Authentication:** Google OAuth
- **Deployment:** Vercel / Local server

---

## 🔑 Environment Variables
Create a `.env` file in the project root with the following keys:

```bash
LLAMA_API_KEY=your_llama_api_key_here
FLASK_SECRET_KEY=your_flask_secret_here
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here

