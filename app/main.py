from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from app.vectorstore import build_vectorstore_from_pdf
import os

app = FastAPI()
vectorstore = None

# ✅ Load your local model with multithreading and streaming
llm = LlamaCpp(
    model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=os.cpu_count(),   # Use all available CPU cores
    streaming=False             # Set to True if you want real-time token streaming
)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    # Save uploaded file to disk
    path = f"documents/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    # Build vector store from PDF
    vectorstore = build_vectorstore_from_pdf(path)

    return {"status": "PDF processed successfully"}

@app.post("/ask")
async def ask_question(query: str):
    global vectorstore

    if vectorstore is None:
        return JSONResponse(status_code=400, content={"error": "Upload a PDF first."})

    # ✅ Retrieve top 2-3 chunks only
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Use `stuff` if you trust the data fits in context window
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # much faster than map_reduce/refine
        return_source_documents=True
    )

    try:
        result = qa_chain.invoke({"query": query})
        return {"answer": result["result"]}
    except ValueError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
