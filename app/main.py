from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path

from app.utils import extract_text_from_pdf
from app.qa_chain import build_qa_chain
from app.vectorstore import build_vectorstore_from_pdf  # ✅ Import function only

from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA

app = FastAPI()
session_store = {}

# Load your local LLM (Mistral, etc.)
llm = LlamaCpp(model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=4096)
vectorstore = None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    contents = await file.read()
    path = f"documents/{file.filename}"
    with open(path, "wb") as f:
        f.write(contents)

    # ✅ Build and store vectorstore
    vectorstore = build_vectorstore_from_pdf(path)

    return {"status": "PDF processed successfully"}

@app.post("/ask")
async def ask_question(query: str):
    global vectorstore

    if vectorstore is None:
        return JSONResponse(status_code=400, content={"error": "Upload a PDF first."})

    # ✅ Limit retrieved chunks to 3 (or even 2 if needed)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        return_source_documents=True
    )

    try:
        result = qa_chain.invoke({"query": query})
        return {"answer": result["result"]}
    except ValueError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})