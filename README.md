
# 🤖 Advanced Medical Chatbot with RAG and Conversational Memory 🧠

A sophisticated, AI-powered medical chatbot that provides accurate, context-aware answers from a knowledge base, featuring conversational memory and a fully automated CI/CD pipeline to Google Cloud Platform.

## 🚀 Features

- **Conversational Interface**: A simple and intuitive web-based chat interface built with Flask.
- **Retrieval-Augmented Generation (RAG)**: Fetches relevant information from a medical PDF knowledge base to ground its answers in facts, reducing hallucinations and improving accuracy.
- **Conversational Memory**: Remembers the context of the conversation to understand and answer follow-up questions intelligently.
- **State-of-the-Art AI**: Powered by Google's advanced Gemini 1.5 Flash model for generation and text-embedding-004 for superior semantic search.
- **Automated CI/CD Pipeline**: Automatically builds, tests, and deploys the application to GCP on every push to the main branch using GitHub Actions.
- **Scalable Infrastructure**: Uses Docker for containerization and Google Cloud Platform for robust, scalable cloud hosting.

## 🛠️ Tech Stack

| Category        | Technology         | Description                                                  |
|----------------|--------------------|-------------------------------------------------------------- |
| Backend         | Python, Flask      | Core programming language and web framework.                 |
| AI Framework    | LangChain          | Framework for developing applications powered by LLMs.       |
| Language Model  | Google Gemini      | gemini-1.5-flash for generation and text-embedding-004.      |
| Vector Database | Pinecone           | Serverless vector DB for efficient similarity search.        |
| Containerization| Docker             | Packages app and dependencies for consistent deployment.     |
| Cloud Platform  | Google Cloud (GCP) | Cloud provider for app hosting.                              |
| ↳ Services      | Compute Engine, Artifact Registry | VM for running container and Docker repo.     |
| CI/CD           | GitHub Actions     | Automates build, push, and deploy pipeline.                  |

## 📂 Project Structure

```
.
├── .github/workflows/
│   └── gcp-deploy.yml      
├── data/
│   └── medical_book.pdf    
├── research/
│   └── trials.ipynb        
├── src/
│   ├── __init__.py
│   ├── helper.py           
│   ├── memory.py           
│   └── prompt.py           
├── templates/
│   └── chat.html           
├── .env                    
├── .gitignore
├── app.py                  
├── Dockerfile              
├── LICENSE
├── README.md               
└── requirements.txt        
```

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/FaheemKhan0817/Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-GCP.git
cd Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-GCP
```

### 2. Create a Conda Environment
```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file and add:
```
GOOGLE_API_KEY="YOUR_GOOGLE_AI_STUDIO_API_KEY"
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
```

### 5. Populate the Vector Database
```bash
python store_index.py
```

### 6. Run the Web Application
```bash
python app.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080)

## ☁️ Deployment (CI/CD with GCP)

1. **GCP Setup**
   - Enable Artifact Registry and Compute Engine APIs.
   - Create Service Account with necessary roles.
   - Create Docker repository.
   - Create VM and open port 8080.

2. **GitHub Setup**
   - Add self-hosted runner.
   - Add secrets: `GCP_PROJECT_ID`, `GCP_SA_KEY`, `GCP_REGION`, `GCP_ARTIFACT_REPO`, `GOOGLE_API_KEY`, `PINECONE_API_KEY`

## 🤝 How to Contribute

1. Fork & Clone
2. Create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit & Push
   ```bash
   git commit -m "feat: Implement amazing new feature"
   git push origin feature/your-feature-name
   ```
4. Open a Pull Request

## 📄 License

Apache-2.0 License. See `LICENSE` file.

---

<p align="center">Made with ❤️ by Faheem Khan</p>

## 🌐 Connect with Me

- 🔗 [LinkedIn](https://linkedin.com/in/faheemkhanml)
- 🌍 [Portfolio](https://www.datascienceportfol.io/Faheem_Khan)

## 🔴 Live Demo
[Click here to try the chatbot](http://34.61.240.234:8080/)

## 📢 Why This Project Stands Out

- Built with production-grade tools (LangChain, GCP, Pinecone, Docker).
- Demonstrates real-world CI/CD pipelines using GitHub Actions and GCP.
- Highlights advanced concepts like conversational memory and retrieval augmentation.
- Ready for deployment, making it ideal for startup MVPs or healthcare tech demos.

## 🎥 Demo

Here’s a quick walkthrough of the chatbot in action:

![Chatbot Demo](assets/chatbot-demo.gif)
