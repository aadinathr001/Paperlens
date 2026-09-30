import os
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

from document_processor import validate_file, process_file, MAX_FILES
from rag import build_faiss_index, run_rag

load_dotenv()

app = FastAPI()

# Mount Static and Template Folders
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global In-Memory Session Store
session = {
    "faiss_index": None,
    "metadata": [],
    "processed_files": [],
}

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def serve_home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/process")
async def process_documents(files: List[UploadFile] = File(...)):
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} files allowed.")

    all_chunks = []
    success_names = []

    for file in files:
        contents = await file.read()
        err = validate_file(file.filename, len(contents))
        if err:
            raise HTTPException(status_code=400, detail=err)
        
        chunks = process_file(contents, file.filename)
        all_chunks.extend(chunks)
        success_names.append(file.filename)

    index, metadata = build_faiss_index(all_chunks)
    session["faiss_index"] = index
    session["metadata"] = metadata
    session["processed_files"] = success_names

    return {
        "status": "success",
        "chunks": len(all_chunks),
        "files_count": len(success_names),
        "files": success_names
    }

@app.post("/api/query")
def ask_question(payload: QueryRequest):
    if not session["faiss_index"]:
        raise HTTPException(status_code=400, detail="No index found. Upload files first.")

    try:
        answer, chunks = run_rag(
            question=payload.question,
            index=session["faiss_index"],
            metadata=session["metadata"],
            groq_api_key=os.environ.get("GROQ_API_KEY")
        )
        return {"answer": answer, "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
def reset():
    session["faiss_index"] = None
    session["metadata"] = []
    session["processed_files"] = []
    return {"status": "cleared"}