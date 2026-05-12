# from dotenv import load_dotenv
# import os

# load_dotenv()

# # Load the pdf file and split the text into chunks
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # Create langchain embedding and vector database
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_community.vectorstores import FAISS

# # Create ChatPrompt template to fed to LLM
# from langchain_core.prompts import ChatPromptTemplate


# # Using GrogCloud makes it much easier to deploy on free providers because llms requires a lot of RAM
# from langchain_groq import ChatGroq


# # # Download the HuggingFace transformer pipeline from the quantized LLM
# # from transformers import AutoTokenizer, AutoModelForCausalLM
# # from transformers import pipeline, BitsAndBytesConfig

# # # Convert the "HuggingFace transformer pipeline" to the "HuggingFace langchain pipeline" of quantized LLM model
# # from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# # As this is a QA task for langchain
# from langchain_classic.chains.retrieval_qa.base import RetrievalQA



# class Search_RAG_LLM:
#     def __init__(self, file_path=None, device=None, faiss_path='faiss_index'):
#         self.faiss_path = faiss_path
        
#         if not os.path.exists(self.faiss_path):
#             if file_path:
#                 file = PyPDFLoader(file_path)
#                 pages = file.load_and_split()
#                 text_splitter = RecursiveCharacterTextSplitter(
#                     chunk_size=1024,
#                     chunk_overlap=256
#                 )
#                 self.chunks = text_splitter.split_documents(pages)


#         if device:
#             self.device = device
#         else:
#             self.device = 'cpu'

#         self.embedding = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': self.device})

#         if not os.path.exists(self.faiss_path):
#             self._create_vector_database()

#         self.vector_db = self._load_vector_database()

#         print("Loading QA chain...")

#         self.qa_chain = self.load_model(k=3)

#         print("Model loaded successfully")

#     def _create_vector_database(self):
#         vector_db = FAISS.from_texts(
#             [chunk.page_content for chunk in self.chunks],
#             embedding=self.embedding
#         )

#         # Save the FAISS database locally
#         vector_db.save_local(folder_path=self.faiss_path, index_name='my_faiss_index')
    
#     def _load_vector_database(self):
#         vector_db = FAISS.load_local(
#             folder_path=self.faiss_path,
#             index_name='my_faiss_index',
#             embeddings=self.embedding,
#             allow_dangerous_deserialization=True
#         )
#         return vector_db
    
#     def load_model(self, k):
#         prompt = """
#         # [INST]
#         You are a helpful assistant.

#         Use the following context to answer the question.

#         Context:
#         {context}

#         Question:
#         {question}
#         # [/INST]
#         """

#         prompt = ChatPromptTemplate.from_template(prompt)
        
#         # bnb_config = BitsAndBytesConfig(
#         #     load_in_4bit=True,
#         #     bnb_4bit_quant_type="nf4",
#         # )

#         # print("Loading tokenizer...")
#         # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
#         # print("Tokenizer loaded")

#         # print("Loading model...")
#         # model = AutoModelForCausalLM.from_pretrained(
#         #     "mistralai/Mistral-7B-Instruct-v0.2",
#         #     quantization_config=bnb_config,
#         #     # device_map="auto"
#         # )
#         # print("Model loaded")

#         # pipe = pipeline(
#         #     "text-generation",
#         #     model=model,
#         #     tokenizer=tokenizer,
#         #     max_new_tokens=1000,
#         # )

#         # lc_pipeline = HuggingFacePipeline(pipeline=pipe)

#         llm = ChatGroq(
#             groq_api_key=os.getenv('GROQ_API_KEY'),
#             model_name="llama-3.3-70b-versatile"
#         )

#         print("Loading qa_chain...")
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,# llm=lc_pipeline,
#             retriever=self.vector_db.as_retriever(search_kwargs={"k": k}),
#             return_source_documents=True,
#             chain_type_kwargs={"prompt": prompt}
#         )
#         print("QA_chain loaded")
#         return qa_chain
    
#     def get_result(self, question):
#         result = self.qa_chain.invoke({"query": question})
#         return result



# As I am using a free tier hosting provider I have to get everything
# (Embedding, LLM) from API and remove heavy parts like faiss-cpu torck, sentence-transformer
import os
from pypdf import PdfReader
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


class RAGService:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )

        self.chunks = self._load_and_split_pdf()

    def _load_and_split_pdf(self):
        reader = PdfReader(self.pdf_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        return text.split("\n\n")

    def _get_relevant_chunks(self, question: str, k: int = 4):
        question_words = set(question.lower().split())
        scored = []

        for chunk in self.chunks:
            chunk_lower = chunk.lower()
            score = sum(1 for w in question_words if w in chunk_lower)
            scored.append((score, chunk))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [c for _, c in scored[:k]]

    def ask(self, question: str):
        context = "\n\n".join(self._get_relevant_chunks(question))

        prompt = f"""
        You are a helpful assistant.

        Use the context below to answer the question.

        Context:
        {context}

        Question:
        {question}
        """

        response = self.llm.invoke(prompt)
        return response.content