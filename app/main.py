from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from app.vectorstore import build_vectorstore_from_pdf
from app.utils import extract_text_from_pdf
import os,json, shutil
from app.prompt import template_dict
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
vectorstore = None

llm = LlamaCpp(
    model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=os.cpu_count(),   
    streaming=False             
)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file extension. Please upload a PDF.")

    path = f"documents/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    vectorstore = build_vectorstore_from_pdf(path)
    return {"status": "PDF processed successfully"}

@app.post("/ask")
async def ask_question(query: str):
    global vectorstore

    if vectorstore is None:
        return JSONResponse(status_code=400, content={"error": "Upload a PDF first."})

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True)

    try:
        # result_dict = {}
        # for key,question in template_dict().items():
        #     prompt = f"""
        #         You are an intelligent assistant. Read the following Form 16 content and extract the required information.

        #         Question:
        #         {question["question"]}

        #         Instruction:
        #         {question["answer"]}

        #         Strictly return only the answer with no explanation.
        #         """

        #     result = qa_chain.invoke({"query": prompt})
        #     result_dict[key] = result["result"]
        # # return {"answer": result["result"]}
        result = qa_chain.invoke({"query": query})
        result = result["result"]
        return {"answer": result}
    except ValueError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def truncate_context(text: str, max_tokens: int = 3000) -> str:
    """Truncate text to stay within model context window (4096 tokens)"""
    max_chars = max_tokens * 4  # 1 token ≈ 4 chars
    return text[:max_chars]

@app.post("/upload_and_extract")
async def upload_and_extract(file: UploadFile = File(...)):
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract and truncate context
    raw_context = extract_text_from_pdf(file_path)
    context = truncate_context(raw_context, max_tokens=3500)

    # Build prompt
    template_json = json.dumps(template_dict(), indent=2)
    prompt_template = PromptTemplate(
    input_variables=["context", "template"],
    template="""
            You are an expert at reading and understanding Form 16 PDF documents.
            Use the following extracted text from a Form 16 document and extract the relevant fields and fill the following JSON format accordingly.

            If the data is not found, leave the field value as null.

            <context>
            {context}
            </context>

            Expected output format:
            {template}
            """,
            )
    formatted_prompt = prompt_template.format(
        context=context,
        template=template_json
    )

    llm = LlamaCpp(
            model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            temperature=0.1,
            max_tokens=512,
            top_p=0.95,
            n_ctx=4096,
            verbose=True,
                )

    question_dict = template_dict()
    final_result = {}

    for key, qa in question_dict.items():
        prompt = f"""
        You are an intelligent assistant. Read the following Form 16 content and extract the required information.

        Document:
        {extract_text_from_pdf(file_path)}

        Question:
        {qa["question"]}

        Instruction:
        {qa["answer"]}

        Strictly return only the answer with no explanation.
        """
        try:
            print(f"\nPrompt length (chars): {len(prompt)} | Key: {key}")
            answer = llm.invoke(prompt).strip()
            final_result[key] = answer
        except Exception as e:
            final_result[key] = f"Error: {str(e)}"

    return JSONResponse(content=final_result)