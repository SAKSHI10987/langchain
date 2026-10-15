# YouTube Chatbot using LangChain and RAG 

A simple AI-powered chatbot that answers questions about YouTube videos using their transcripts.

This project uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from a video transcript and generate answers using an LLM.

#Technologies Used

- Python
- LangChain
- YouTube Transcript API
- Hugging Face Embeddings
- FAISS Vector Store
- Groq LLM
- Streamlit

#How It Works

1. Enter a YouTube video ID.
2. Fetch the English transcript.
3. Split the transcript into smaller chunks.
4. Convert the chunks into embeddings using Hugging Face.
5. Store and retrieve relevant chunks using FAISS.
6. Generate answers using LangChain and Groq.
7. Display the answer in Streamlit.

#Installation

Install the required libraries:

```bash
pip install streamlit langchain-huggingface langchain-core langchain-classic langchain-groq youtube-transcript-api faiss-cpu sentence-transformers python-dotenv
```

#Setup

Create a `.env` file and add your Groq API key:

```env
GROQ_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run app.py
```

#Limitations

- Currently supports English transcripts only.
- Answers may not always be accurate.
- Some videos may not have accessible transcripts.
- Further improvements are needed.

#Future Improvements

- Support multiple languages.
- Improve answer accuracy.
- Improve error handling and retrieval.

#Learning Outcomes

This project helped me understand RAG, text splitting, embeddings, vector databases, LangChain pipelines, and LLM integration.

---

**Built as part of my journey into LangChain and Generative AI.** 🚀
