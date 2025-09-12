# LLaMA-Reader

## Installation 

Follow these steps to set up LLaMA-Reader locally:

### Requirements 
- Python 3.8 or higher
- Git 
- pip 

### Steps 

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Doc2Question.git
cd Doc2Question
```

2. **Create the environment:**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Run app:**
```bash
python app.py
```

## Key Technologies

This project is built on top of the following main technologies:

- **LangChain** – Framework for building LLM-powered applications  
- **Hugging Face Transformers** – Pre-trained models (google/flan-t5-large) for question generation  
- **Sentence-Transformers** – Embeddings (all-mpnet-base-v2) for semantic search  
- **FAISS / ChromaDB** – Vector databases for storing and retrieving embeddings  
- **FastAPI + Uvicorn** – Web framework and server for the user interface  
- **PyPDF / PyPDF2** – PDF parsing and text extraction  

<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTGGDRzWjTYrD5HSvHziGN6t6UcVDZVIwq2rw&s" alt="LangChain" width="120"/>
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="120"/>
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR23iOonB187tJq9_-Zo4Vt2I8rjvnFwYd-Eg&s" alt="Sentence Transformers" width="120"/>
  <img src="https://daxg39y63pxwu.cloudfront.net/images/blog/faiss-vector-database/FAISS_Vector_Database.webp" alt="FAISS" width="120"/>
  <img src="https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png" alt="FastAPI" width="120"/>
</p>
