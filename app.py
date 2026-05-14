# import streamlit as st
# from rag_vector_db_llm import Search_RAG_LLM
# import warnings
# import logging
# import os
# import torch
# import re

# warnings.filterwarnings("ignore")

# logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# device = "cuda" if torch.cuda.is_available() else "cpu"

# path = os.path.join(os.getcwd(), '1706.03762v7.pdf')


# @st.cache_resource
# def initialize():

#     st.write("Initializing class...")

#     if os.path.exists("./faiss_index"):
#         obj = Search_RAG_LLM(device=device)
#     else:
#         obj = Search_RAG_LLM(path, device=device)

#     return obj



# st.title("Ask questions related to this PDF")

# search_obj = initialize()

# question = st.text_input("Enter the question")

# if st.button("Generate"):

#     if question.strip():

#         with st.spinner("Generating answer..."):

#             result = search_obj.get_result(
#                 question
#             )

#             answer = re.sub(
#                 '\n',
#                 '',
#                 result['result'].split('[/INST]')[1]
#             )

#             st.write(answer)

#     else:
#         st.warning("Please enter a question.")






# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from rag_vector_db_llm import Search_RAG_LLM
# import warnings
# import logging
# import os
# import torch
# import re

# warnings.filterwarnings("ignore")

# logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI(title="RAG PDF QA API")

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # or ["http://127.0.0.1:5500"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# device = "cuda" if torch.cuda.is_available() else "cpu"

# path = os.path.join(os.getcwd(), "1706.03762v7.pdf")

# search_obj = None


# def load_model():

#     global search_obj
#     global qa_chain

#     if os.path.exists("./faiss_index"):
#         search_obj = Search_RAG_LLM(device=device)
#     else:
#         search_obj = Search_RAG_LLM(path, device=device)


# @app.on_event("startup")
# def startup_event():
#     load_model()


# class QuestionRequest(BaseModel):
#     question: str


# @app.get("/")
# def home():
#     return {
#         "message": "RAG PDF Question Answering API is running"
#     }


# @app.post("/search")
# def ask_question(request: QuestionRequest):

#     question = request.question.strip()

#     if not question:
#         return {
#             "error": "Question cannot be empty"
#         }

#     result = search_obj.get_result(question)

#     return {
#         "question": question,
#         "answer": re.sub('\n', '', result['result'])
#         # "answer": re.sub('\n', '', result['result'].split('[/INST]')[1])
#     }



from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os

from rag_vector_db_llm import RAGService

app = FastAPI(title="Groq RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pdf_path = os.path.join(
    os.getcwd(),
    "1706.03762v7.pdf"
)

@app.on_event("startup")
async def startup_event():

    global rag

    print("Initializing RAG service...")

    app.state.rag = RAGService(pdf_path)

    print("RAG service initialized")

app.state.rag = RAGService(pdf_path)

class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():

    return {
        "message": "Groq RAG API running"
    }


@app.post("/search")
async def search(request: QuestionRequest):

    question = request.question.strip()

    if not question:

        return {
            "error": "Question cannot be empty"
        }

    answer = await app.state.rag.ask(question)

    return {
        "question": question,
        "answer": answer
    }
