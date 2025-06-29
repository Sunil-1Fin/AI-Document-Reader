from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.llms import LlamaCpp

import os

def build_qa_chain(doc_text: str):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(doc_text)]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)

    llm = LlamaCpp(
        model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.7,
        max_tokens=512,
        n_gpu_layers=20,
        n_batch=128,
        top_p=0.95,
        n_ctx=4096,
        verbose=True,
    )

    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="map_reduce",  # or "refine"
        return_source_documents=False,  # Only return the final result
    )
    return qa