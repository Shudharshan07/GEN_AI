import os
import json
import uuid
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime, date, timedelta
from pathlib import Path as PythonPath
import io

import faiss
import numpy as np
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Path as FastApiPath
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from gtts import gTTS

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
DATA_DIR = "app_data"
UPLOAD_DIR = "uploads"
VECTOR_DB_DIR = "vector_db"
STUDY_GUIDE_DIR = "study_guides"
NOTES_DIR = "generated_notes" # ADDED: For AI-generated notes
DOUBT_LOG_FILE = os.path.join(DATA_DIR, "doubt_log.json")
SESSION_FILE = os.path.join(DATA_DIR, "study_session.json")

CHUNK_SIZE = 500  # words
CHUNK_OVERLAP = 50  # words
TOP_K_RETRIEVAL = 5

# Ensure all necessary directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STUDY_GUIDE_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True) # ADDED: Create notes directory

# --- Initialize External Services ---
# Initialize Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the Sentence Transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Automated PDF Study Guide Generator",
    description="V4.6.0 - PDF Q&A with AI Note Generation, TTS Audio, and Study Tools.",
    version="4.6.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Data Models for Data Structures ---

# This class represents a single question-answer record. It can also be flagged as a "doubt".
class QARecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    is_unclear: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)


# --- Helper Classes for State Management ---

class ResponseLog:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.records: Dict[str, QARecord] = self._load()

    def _load(self) -> Dict[str, QARecord]:
        if not os.path.exists(self.filepath):
            return {}
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {rec_id: QARecord(**rec_data) for rec_id, rec_data in data.items()}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _save(self):
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump({rec_id: rec.dict() for rec_id, rec in self.records.items()}, f, indent=2, default=str)

    def add(self, record: QARecord):
        self.records[record.id] = record
        self._save()

    def get(self, record_id: str) -> Optional[QARecord]:
        return self.records.get(record_id)

    def get_all(self, unclear_only: bool = False) -> List[QARecord]:
        records_list = sorted(self.records.values(), key=lambda r: r.timestamp, reverse=True)
        if unclear_only:
            return [r for r in records_list if r.is_unclear]
        return records_list

    def mark_as_unclear(self, record_id: str, status: bool = True) -> Optional[QARecord]:
        record = self.get(record_id)
        if record:
            record.is_unclear = status
            self._save()
            return record
        return None

class StudySession:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.last_activity_date: Optional[date] = None
        self.streak: int = 0
        self.ongoing_topics: List[str] = []
        self._load()

    def _load(self):
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                last_date_str = data.get("last_activity_date")
                if last_date_str:
                    self.last_activity_date = date.fromisoformat(last_date_str)
                self.streak = data.get("streak", 0)
                self.ongoing_topics = data.get("ongoing_topics", [])
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    def _save(self):
        data = {
            "last_activity_date": self.last_activity_date.isoformat() if self.last_activity_date else None,
            "streak": self.streak,
            "ongoing_topics": self.ongoing_topics
        }
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def log_activity(self, topic: str):
        today = date.today()
        if self.last_activity_date:
            if today == self.last_activity_date + timedelta(days=1):
                self.streak += 1
            elif today != self.last_activity_date:
                self.streak = 1
        else:
            self.streak = 1
        
        self.last_activity_date = today

        if topic not in self.ongoing_topics:
            self.ongoing_topics.insert(0, topic)
        self.ongoing_topics = self.ongoing_topics[:5]

        self._save()

    def get_motivation(self) -> Dict[str, Any]:
        badge = "🌱 Budding Learner"
        if self.streak >= 3: badge = "🔥 On Fire!"
        if self.streak >= 7: badge = "🧠 Scholar"
        if self.streak >= 30: badge = "👑 Mastery"
        
        return {
            "current_streak_days": self.streak,
            "badge": badge,
            "last_activity": self.last_activity_date
        }

# --- Core Logic Classes ---

class DocumentChunk:
    def __init__(self, text: str, filename: str, page_num: int, chunk_id: int):
        self.text = text
        self.filename = filename
        self.page_num = page_num
        self.chunk_id = chunk_id
        self.embedding: Optional[np.ndarray] = None

class VectorDatabase:
    def __init__(self):
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.dimension = 384
        
    def initialize_index(self):
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        self.initialize_index()
        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_model.encode(texts, normalize_embeddings=True)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[tuple]:
        if self.index is None or not self.chunks:
            return []
        query_embedding = embedding_model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.chunks)))
        return [(self.chunks[idx], float(score)) for score, idx in zip(scores[0], indices[0]) if idx != -1]
    
    def save_to_disk(self, filepath: str):
        if self.index:
            faiss.write_index(self.index, f"{filepath}.faiss")
        chunks_data = [{
            'text': chunk.text, 'filename': chunk.filename, 'page_num': chunk.page_num,
            'chunk_id': chunk.chunk_id,
            'embedding': chunk.embedding.tolist() if chunk.embedding is not None else None
        } for chunk in self.chunks]
        with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    def load_from_disk(self, filepath: str):
        try:
            if os.path.exists(f"{filepath}.faiss"):
                self.index = faiss.read_index(f"{filepath}.faiss")
            if os.path.exists(f"{filepath}.json"):
                with open(f"{filepath}.json", 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                self.chunks = []
                for data in chunks_data:
                    chunk = DocumentChunk(text=data['text'], filename=data['filename'], page_num=data['page_num'], chunk_id=data['chunk_id'])
                    if data.get('embedding'):
                        chunk.embedding = np.array(data['embedding'])
                    self.chunks.append(chunk)
                return True
        except Exception as e:
            print(f"Error loading vector database from disk: {e}")
        return False
    
    def clear(self):
        self.index = None
        self.chunks = []

class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> Dict[int, str]:
        page_texts = {}
        try:
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    page_texts[page_num + 1] = text
            doc.close()
        except Exception as e:
            raise Exception(f"Error reading PDF file {file_path}: {str(e)}")
        return page_texts

    @staticmethod
    def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunks.append(' '.join(words[start:end]))
            if end >= len(words):
                break
            start = end - overlap
        return chunks

class GeminiService:
    @staticmethod
    def _get_model(model_name: str = "gemini-1.5-flash"):
        return genai.GenerativeModel(model_name)

    @staticmethod
    def generate_answer(query: str, relevant_chunks: List[tuple], mode: str = "standard") -> Dict[str, Any]:
        if not relevant_chunks:
            return {"answer": "I couldn't find relevant information in the uploaded documents to answer your question.", "sources": [], "confidence": 0.0}
        
        context_parts, sources = [], []
        for i, (chunk, score) in enumerate(relevant_chunks, 1):
            context_parts.append(f"[Source {i}] From file '{chunk.filename}', Page {chunk.page_num}:\n{chunk.text}")
            sources.append({
                "source_id": i, "filename": chunk.filename, "page": chunk.page_num,
                "relevance_score": round(score, 3), "preview": chunk.text[:150] + "..."
            })
        
        context = "\n\n".join(context_parts)
        
        prompts = {
            "speed": f"""You are in "Speed Mode". Provide an ultra-short, one-sentence answer to the question based ONLY on the provided excerpts. Use markdown formatting for clarity. Cite sources like [Source X].

Question: {query}

Excerpts:
{context}

Answer (use markdown formatting):""",
            "deep_dive": f"""You are in "Deep Dive Mode". Provide a detailed, step-by-step explanation for the question using ONLY the provided excerpts. Use proper markdown formatting including:
- **Bold** for key concepts
- *Italics* for emphasis
- `code` for technical terms
- ## Headers for main sections
- ### Subheaders for subsections
- - Bullet points for lists
- 1. Numbered lists for steps
- > Blockquotes for important notes

Where possible, use examples or analogies to clarify complex points. Be comprehensive and well-structured. Cite sources like [Source X] for every piece of information.

Question: {query}

Excerpts:
{context}

Detailed Answer (use markdown formatting):""",
            "standard": f"""You are an expert document analyst. Answer the following question based ONLY on the provided document excerpts. Use clean markdown formatting to structure your response:
- Use **bold** for important terms
- Use *italics* for emphasis
- Use bullet points (-) for lists
- Use numbered lists (1.) for steps
- Use > for important quotes or notes
- Use ## for main sections if needed

You must cite the sources you use in your answer using the format [Source X].

Question: {query}

Excerpts:
{context}

Answer (use markdown formatting):"""
        }
        prompt = prompts.get(mode, prompts["standard"])

        try:
            model = GeminiService._get_model()
            response = model.generate_content(prompt)
            avg_score = np.mean([score for _, score in relevant_chunks])
            return {
                "answer": response.text.strip(), "sources": sources,
                "confidence": min(avg_score * 1.2, 1.0)
            }
        except Exception as e:
            raise Exception(f"Error communicating with the Gemini API: {str(e)}")

    @staticmethod
    def generate_full_study_guide(full_text: str, topic: str) -> Dict[str, Any]:
        prompt = f"""
You are an expert educator. Create a comprehensive study guide for a document about "{topic}".
The guide must have "roadmap", "flashcards", and "quiz" sections in a single JSON object.
- Roadmap: 3-5 logical steps with "step_title" and "description" (use markdown formatting in descriptions with **bold** for key concepts and *italics* for emphasis).
- Flashcards: 5-10 key terms with "question" and "answer" (use markdown formatting in answers for better readability).
- Quiz: 3-5 multiple-choice questions with "question_text", "options" list, and "correct_answer".

Use markdown formatting in the roadmap descriptions and flashcard answers to make them more structured and readable.

Based only on the text below:
--- DOCUMENT TEXT START ---
{full_text}
--- DOCUMENT TEXT END ---
"""
        model_response_text = ""
        try:
            model = GeminiService._get_model("gemini-1.5-flash")
            response = model.generate_content(prompt)
            model_response_text = response.text
            clean_response = model_response_text.strip().replace("```json", "").replace("```", "")
            return json.loads(clean_response)
        except (Exception, json.JSONDecodeError) as e:
            error_msg = f"Error generating study guide: {e}. Response started with: {model_response_text[:200]}"
            raise Exception(error_msg)

    # ADDED: New method for generating notes
    @staticmethod
    def generate_notes_from_text(full_text: str, topic: str) -> Dict[str, Any]:
        prompt = f"""
You are an expert academic assistant. Your task is to read the following document about "{topic}" and generate a concise set of study notes in a structured JSON format.

The JSON object you create must contain these exact keys:
1.  "topic": A brief, accurate title for the notes, derived from the document's content.
2.  "summary": A 2-3 sentence high-level summary of the entire document (use markdown formatting).
3.  "notes": A list of 5-7 key points. Each point in the list must be a JSON object with:
    - "title": A short, clear heading for the key point.
    - "details": A paragraph explaining the concept, its significance, and any important details (use markdown formatting with **bold** for key terms, *italics* for emphasis, and `code` for technical terms).
    - "keywords": A list of 3-5 essential keywords or terms related to this point.

Use markdown formatting in the summary and details fields to make the content more readable and structured.

Generate the response based ONLY on the provided text. Ensure the output is a single, valid JSON object.

--- DOCUMENT TEXT START ---
{full_text}
--- DOCUMENT TEXT END ---
"""
        model_response_text = ""
        try:
            model = GeminiService._get_model("gemini-1.5-flash")
            response = model.generate_content(prompt)
            model_response_text = response.text
            clean_response = model_response_text.strip().replace("```json", "").replace("```", "")
            return json.loads(clean_response)
        except (Exception, json.JSONDecodeError) as e:
            error_msg = f"Error generating notes: {e}. Response started with: {model_response_text[:200]}"
            raise Exception(error_msg)


# --- Global Instances ---
vector_db = VectorDatabase()
vector_db_path = os.path.join(VECTOR_DB_DIR, "main_db")
vector_db.load_from_disk(vector_db_path)

response_log = ResponseLog(DOUBT_LOG_FILE)
study_session = StudySession(SESSION_FILE)


# --- API Request/Response Models ---
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = TOP_K_RETRIEVAL
    mode: Optional[Literal["standard", "speed", "deep_dive"]] = "standard"

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    response_id: str
    mode: str
    tts_endpoint: str

class UploadResponse(BaseModel):
    message: str
    files_processed: List[Dict]

class StudyGuideResponse(BaseModel):
    topic: str
    roadmap: List[Dict[str, str]]
    flashcards: List[Dict[str, str]]
    quiz: List[Dict[str, Any]]

# ADDED: Response model for generated notes
class NoteItem(BaseModel):
    title: str
    details: str
    keywords: List[str]

class GeneratedNotesResponse(BaseModel):
    topic: str
    summary: str
    notes: List[NoteItem]

class DatabaseStats(BaseModel):
    total_chunks: int
    total_documents: int
    documents: List[Dict[str, Any]]


# --- API Endpoints ---

@app.post("/upload", response_model=UploadResponse, tags=["1. Document Management"])
async def upload_pdfs(files: List[UploadFile] = File(...)):
    processed_files, all_chunks = [], []
    current_chunk_count = len(vector_db.chunks)
    try:
        for file in files:
            if not file.filename.endswith('.pdf'):
                continue
            
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            page_texts = PDFProcessor.extract_text_from_pdf(file_path)
            file_chunks = []
            for page_num, text in page_texts.items():
                chunks_from_page = PDFProcessor.chunk_text(text)
                for i, chunk_text in enumerate(chunks_from_page):
                    chunk_id = current_chunk_count + len(all_chunks) + len(file_chunks)
                    file_chunks.append(DocumentChunk(text=chunk_text, filename=file.filename, page_num=page_num, chunk_id=chunk_id))
            
            all_chunks.extend(file_chunks)
            processed_files.append({"filename": file.filename, "pages": len(page_texts), "chunks_added": len(file_chunks)})
        
        if all_chunks:
            vector_db.add_chunks(all_chunks)
            vector_db.save_to_disk(vector_db_path)
            
        return UploadResponse(message=f"{len(processed_files)} PDF(s) processed and indexed.", files_processed=processed_files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during file upload: {e}")

@app.post("/generate_study_guide/{filename}", response_model=StudyGuideResponse, tags=["2. Automated Learning Tools"])
async def generate_study_guide(filename: str = FastApiPath(..., description="The exact filename of a previously uploaded PDF.")):
    guide_path = os.path.join(STUDY_GUIDE_DIR, f"{filename}.json")

    if os.path.exists(guide_path):
        with open(guide_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    doc_chunks = [chunk for chunk in vector_db.chunks if chunk.filename == filename]
    if not doc_chunks:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found. Please upload it via the /upload endpoint first.")

    full_text = "\n\n".join([chunk.text for chunk in sorted(doc_chunks, key=lambda c: c.page_num)])
    topic = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()

    try:
        study_guide_data = GeminiService.generate_full_study_guide(full_text, topic)
        
        response_data = StudyGuideResponse(
            topic=topic,
            roadmap=study_guide_data.get("roadmap", []),
            flashcards=study_guide_data.get("flashcards", []),
            quiz=study_guide_data.get("quiz", [])
        )

        with open(guide_path, 'w', encoding='utf-8') as f:
            json.dump(response_data.dict(), f, indent=2)
        
        study_session.log_activity(f"Generated guide for {topic}")
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ADDED: New endpoint for generating notes
@app.post("/generate_notes/{filename}", response_model=GeneratedNotesResponse, tags=["2. Automated Learning Tools"])
async def generate_ai_notes(filename: str = FastApiPath(..., description="The filename of the PDF to generate notes from.")):
    """
    Generates a structured set of notes from the content of a specified PDF.
    Caches results to avoid re-generation.
    """
    notes_path = os.path.join(NOTES_DIR, f"{filename}.json")

    # 1. Check for cached notes
    if os.path.exists(notes_path):
        with open(notes_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # 2. If not cached, find the document chunks
    doc_chunks = [chunk for chunk in vector_db.chunks if chunk.filename == filename]
    if not doc_chunks:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found. Please upload it via the /upload endpoint first.")

    # 3. Assemble full text and generate notes
    full_text = "\n\n".join([chunk.text for chunk in sorted(doc_chunks, key=lambda c: c.page_num)])
    topic = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()

    try:
        notes_data = GeminiService.generate_notes_from_text(full_text, topic)
        
        # 4. Validate and structure the response
        response_data = GeneratedNotesResponse(
            topic=notes_data.get("topic", topic),
            summary=notes_data.get("summary", ""),
            notes=[NoteItem(**note) for note in notes_data.get("notes", [])]
        )

        # 5. Cache the generated notes
        with open(notes_path, 'w', encoding='utf-8') as f:
            json.dump(response_data.dict(), f, indent=2)
        
        # 6. Log the activity
        study_session.log_activity(f"Generated notes for {topic}")
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AnswerResponse, tags=["3. Core Q&A"])
async def ask_question(request: QuestionRequest):
    """
    Ask a question, get an answer, and a corresponding TTS audio endpoint.
    """
    try:
        if not vector_db.chunks:
            raise HTTPException(status_code=400, detail="No documents have been uploaded yet.")
        
        relevant_chunks = vector_db.search(request.question, request.top_k)
        result = GeminiService.generate_answer(request.question, relevant_chunks, request.mode)
        
        new_record = QARecord(question=request.question, answer=result["answer"], sources=result["sources"])
        response_log.add(new_record)
        study_session.log_activity(f"Asked ({request.mode} mode): {request.question[:40]}...")
        
        tts_url = f"/tts/{new_record.id}"
        
        return AnswerResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            response_id=new_record.id,
            mode=request.mode,
            tts_endpoint=tts_url
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.get("/tts/{response_id}", tags=["3. Core Q&A"], response_class=StreamingResponse)
async def text_to_speech(response_id: str = FastApiPath(..., description="The ID of the response to read aloud.")):
    """
    Generates and streams TTS audio for a given response's answer.
    """
    record = response_log.get(response_id)
    if not record:
        raise HTTPException(status_code=404, detail="Response record not found, cannot generate audio.")

    answer_text = record.answer
    if not answer_text.strip():
        raise HTTPException(status_code=400, detail="The answer is empty, nothing to read.")

    try:
        mp3_fp = io.BytesIO()
        tts = gTTS(text=answer_text, lang='en', slow=False)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return StreamingResponse(mp3_fp, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate TTS audio: {str(e)}")

@app.get("/review/all", response_model=List[QARecord], tags=["4. Review & History"])
async def get_all_responses():
    """Get a history of all past questions and answers."""
    return response_log.get_all()

@app.get("/review/unclear", response_model=List[QARecord], tags=["4. Review & History"])
async def get_unclear_responses():
    """Get all responses that have been marked as unclear for review."""
    return response_log.get_all(unclear_only=True)

@app.patch("/review/{response_id}/mark_unclear", response_model=QARecord, tags=["4. Review & History"])
async def mark_response_as_unclear(response_id: str = FastApiPath(..., description="The ID of the response to mark as unclear.")):
    """Flag a response as 'unclear' for later review."""
    updated_record = response_log.mark_as_unclear(response_id, status=True)
    if not updated_record:
        raise HTTPException(status_code=404, detail="Response record not found")
    return updated_record

@app.patch("/review/{response_id}/resolve", response_model=QARecord, tags=["4. Review & History"])
async def resolve_unclear_response(response_id: str = FastApiPath(..., description="The ID of the response to mark as resolved.")):
    """Remove the 'unclear' flag from a response."""
    updated_record = response_log.mark_as_unclear(response_id, status=False)
    if not updated_record:
        raise HTTPException(status_code=404, detail="Response record not found")
    return updated_record

@app.get("/session/stats", tags=["5. Session & Motivation"])
async def get_session_stats():
    return study_session.get_motivation()


@app.get("/session/ongoing_topics", response_model=List[str], tags=["5. Session & Motivation"])
async def get_ongoing_topics():
    return study_session.ongoing_topics


@app.get("/stats", response_model=DatabaseStats, tags=["6. Utility"])
async def get_database_stats():
    documents = {}
    for chunk in vector_db.chunks:
        if chunk.filename not in documents:
            documents[chunk.filename] = {"total_chunks": 0, "pages": set()}
        documents[chunk.filename]["total_chunks"] += 1
        documents[chunk.filename]["pages"].add(chunk.page_num)
    
    doc_list = [{"filename": fn, "total_chunks": data["total_chunks"], "total_pages": len(data["pages"])} for fn, data in documents.items()]
    
    return DatabaseStats(
        total_chunks=len(vector_db.chunks),
        total_documents=len(documents),
        documents=doc_list
    )

@app.delete("/clear", tags=["6. Utility"])
async def clear_all_data():
    """Deletes all uploaded data, response logs, and generated content."""
    try:
        vector_db.clear()
        # MODIFIED: Added NOTES_DIR to the list of directories to clear
        for dir_path in [UPLOAD_DIR, VECTOR_DB_DIR, STUDY_GUIDE_DIR, NOTES_DIR, DATA_DIR]:
            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    try:
                        os.remove(os.path.join(dir_path, filename))
                    except OSError:
                        pass # Ignore if it's a directory
        
        # Re-initialize the global instances with empty state
        global response_log, study_session
        response_log = ResponseLog(DOUBT_LOG_FILE)
        study_session = StudySession(SESSION_FILE)
        
        return {"message": "All databases, logs, and generated content have been cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", tags=["6. Utility"])
async def health_check():
    return {"status": "ok", "version": app.version, "timestamp": datetime.now().isoformat()}

# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn
    print("--- Starting Automated PDF Study Guide Backend ---")
    print(f"Version: {app.version}")
    print("API documentation will be available at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
