# PDF Q&A Backend with FastAPI, PyMuPDF, and Gemini 2.0 Flash
# Requirements: pip install fastapi uvicorn python-multipart pymupdf sentence-transformers faiss-cpu google-generativeai python-dotenv

import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
UPLOAD_DIR = "uploads"
VECTOR_DB_DIR = "vector_db"
CHUNK_SIZE = 500  # words
CHUNK_OVERLAP = 50  # words
TOP_K_RETRIEVAL = 5

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# FastAPI app
app = FastAPI(
    title="PDF Q&A Backend",
    description="PDF Question-Answering system using PyMuPDF, FAISS, and Gemini 2.0 Flash",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentChunk:
    def __init__(self, text: str, filename: str, page_num: int, chunk_id: int, bbox: Optional[List[float]] = None):
        self.text = text
        self.filename = filename
        self.page_num = page_num
        self.chunk_id = chunk_id
        self.bbox = bbox  # Bounding box coordinates [x0, y0, x1, y1]
        self.embedding = None

class VectorDatabase:
    def __init__(self):
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        
    def initialize_index(self):
        """Initialize FAISS index"""
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add chunks to the vector database"""
        self.initialize_index()
        
        # Generate embeddings for all chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_model.encode(texts, normalize_embeddings=True)
        
        # Store embeddings in chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks
        self.chunks.extend(chunks)
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[tuple]:
        """Search for most relevant chunks"""
        if self.index is None or len(self.chunks) == 0:
            return []
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query], normalize_embeddings=True)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.chunks)))
        
        # Return chunks with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save_to_disk(self, filepath: str):
        """Save vector database to disk"""
        if self.index is not None:
            faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save chunks metadata
        chunks_data = []
        for chunk in self.chunks:
            chunks_data.append({
                'text': chunk.text,
                'filename': chunk.filename,
                'page_num': chunk.page_num,
                'chunk_id': chunk.chunk_id,
                'bbox': chunk.bbox,
                'embedding': chunk.embedding.tolist() if chunk.embedding is not None else None
            })
        
        with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    def load_from_disk(self, filepath: str):
        """Load vector database from disk"""
        try:
            # Load FAISS index
            if os.path.exists(f"{filepath}.faiss"):
                self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load chunks
            if os.path.exists(f"{filepath}.json"):
                with open(f"{filepath}.json", 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                
                self.chunks = []
                for data in chunks_data:
                    chunk = DocumentChunk(
                        text=data['text'],
                        filename=data['filename'],
                        page_num=data['page_num'],
                        chunk_id=data['chunk_id'],
                        bbox=data.get('bbox')
                    )
                    if data['embedding']:
                        chunk.embedding = np.array(data['embedding'])
                    self.chunks.append(chunk)
                
                return True
        except Exception as e:
            print(f"Error loading vector database: {e}")
        
        return False
    
    def clear(self):
        """Clear all data"""
        self.index = None
        self.chunks = []

class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> Dict[int, Dict[str, Any]]:
        """Extract text from PDF using PyMuPDF with enhanced metadata"""
        page_data = {}
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text with position information
                text_dict = page.get_text("dict")
                
                # Extract plain text
                text = page.get_text()
                
                if text.strip():  # Only store non-empty pages
                    # Get page dimensions
                    rect = page.rect
                    
                    page_data[page_num + 1] = {
                        'text': text,
                        'width': rect.width,
                        'height': rect.height,
                        'blocks': text_dict.get('blocks', []),  # For advanced text analysis
                        'word_count': len(text.split())
                    }
            
            doc.close()
            
        except Exception as e:
            raise Exception(f"Error reading PDF {file_path}: {str(e)}")
        
        return page_data
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks with better sentence awareness"""
        # Split into sentences first for better chunking
        sentences = text.replace('\n', ' ').split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return [text]
        
        chunks = []
        current_chunk = ""
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_word_count + sentence_words > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_words = current_chunk.split()[-overlap:]
                current_chunk = ' '.join(overlap_words) + ' ' + sentence
                current_word_count = len(overlap_words) + sentence_words
            else:
                current_chunk += ' ' + sentence
                current_word_count += sentence_words
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Fallback to word-based chunking if sentence-based didn't work well
        if not chunks:
            words = text.split()
            if len(words) <= chunk_size:
                return [text]
            
            chunks = []
            start = 0
            
            while start < len(words):
                end = start + chunk_size
                chunk_words = words[start:end]
                chunks.append(' '.join(chunk_words))
                
                if end >= len(words):
                    break
                
                start = end - overlap
        
        return chunks

class GeminiService:
    @staticmethod
    def generate_answer(query: str, relevant_chunks: List[tuple], model: str = "gemini-2.0-flash-exp") -> Dict[str, Any]:
        """Generate answer using Gemini 2.0 Flash"""
        if not relevant_chunks:
            return {
                "answer": "I couldn't find relevant information in the uploaded documents to answer your question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Prepare context from chunks
        context_parts = []
        sources = []
        
        for i, (chunk, score) in enumerate(relevant_chunks, 1):
            context_parts.append(f"[Source {i}] From {chunk.filename}, Page {chunk.page_num}:\n{chunk.text}")
            sources.append({
                "source_id": i,
                "filename": chunk.filename,
                "page": chunk.page_num,
                "relevance_score": round(score, 3),
                "preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "bbox": chunk.bbox
            })
        
        context = "\n\n".join(context_parts)
        
        # Create system prompt for better responses
        system_prompt = """You are an expert document analyst. Your task is to answer questions based on provided document excerpts with high accuracy and proper citations.

Guidelines:
1. Always cite sources using [Source X] format where X is the source number
2. Be specific and detailed in your answers
3. If information is insufficient, clearly state this
4. Synthesize information from multiple sources when relevant
5. Maintain objectivity and accuracy"""
        
        user_prompt = f"""Question: {query}

Document excerpts:
{context}

Please provide a comprehensive answer with proper source citations."""
        
        try:
            model_instance = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_prompt
            )
            
            response = model_instance.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=1000
                )
            )
            
            answer = response.text.strip()
            
            # Calculate confidence based on relevance scores
            avg_score = np.mean([score for _, score in relevant_chunks])
            confidence = min(avg_score * 1.2, 1.0)  # Scale and cap at 1.0
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": round(confidence, 3),
                "model_used": model
            }
            
        except Exception as e:
            raise Exception(f"Error generating answer with Gemini: {str(e)}")

# Global vector database instance
vector_db = VectorDatabase()

# Load existing vector database if available
vector_db_path = os.path.join(VECTOR_DB_DIR, "main_db")
vector_db.load_from_disk(vector_db_path)

# API Models
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = TOP_K_RETRIEVAL
    model: Optional[str] = "gemini-2.0-flash-exp"

class UploadResponse(BaseModel):
    message: str
    files_processed: List[Dict[str, Any]]
    total_chunks: int
    processing_time: float

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    model_used: str

class DatabaseStats(BaseModel):
    total_chunks: int
    total_documents: int
    documents: List[Dict[str, Any]]

# API Endpoints
@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload and process PDF files"""
    start_time = datetime.now()
    
    if len(files) > 2:
        raise HTTPException(status_code=400, detail="Maximum 2 files allowed")
    
    processed_files = []
    all_chunks = []
    
    try:
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            
            # Save uploaded file
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Extract text and metadata from PDF
            page_data = PDFProcessor.extract_text_from_pdf(file_path)
            
            # Create chunks with enhanced metadata
            chunk_id = len(all_chunks)
            total_pages = len(page_data)
            total_words = sum(data['word_count'] for data in page_data.values())
            
            for page_num, data in page_data.items():
                chunks = PDFProcessor.chunk_text(data['text'])
                
                for chunk_text in chunks:
                    chunk = DocumentChunk(
                        text=chunk_text,
                        filename=file.filename,
                        page_num=page_num,
                        chunk_id=chunk_id
                    )
                    all_chunks.append(chunk)
                    chunk_id += 1
            
            processed_files.append({
                "filename": file.filename,
                "pages": total_pages,
                "total_words": total_words,
                "chunks_created": len([c for c in all_chunks if c.filename == file.filename])
            })
        
        # Add chunks to vector database
        vector_db.add_chunks(all_chunks)
        
        # Save vector database
        vector_db.save_to_disk(vector_db_path)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return UploadResponse(
            message="Files processed successfully",
            files_processed=processed_files,
            total_chunks=len(all_chunks),
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about uploaded documents"""
    start_time = datetime.now()
    
    try:
        # Validate that documents are uploaded
        if not vector_db.chunks:
            raise HTTPException(
                status_code=400, 
                detail="No documents have been uploaded yet. Please upload PDF files first."
            )
        
        # Search for relevant chunks
        relevant_chunks = vector_db.search(request.question, request.top_k)
        
        if not relevant_chunks:
            return AnswerResponse(
                question=request.question,
                answer="No relevant information found in the uploaded documents.",
                sources=[],
                confidence=0.0,
                processing_time=0.0,
                model_used=request.model
            )
        
        # Generate answer using Gemini
        result = GeminiService.generate_answer(request.question, relevant_chunks, request.model)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnswerResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            processing_time=round(processing_time, 3),
            model_used=result["model_used"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "total_chunks": len(vector_db.chunks),
        "total_documents": len(set(chunk.filename for chunk in vector_db.chunks)),
        "timestamp": datetime.now().isoformat(),
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "gemini-2.0-flash-exp"
    }

@app.get("/stats", response_model=DatabaseStats)
async def get_database_stats():
    """Get detailed database statistics"""
    documents = {}
    
    for chunk in vector_db.chunks:
        if chunk.filename not in documents:
            documents[chunk.filename] = {
                "filename": chunk.filename,
                "total_chunks": 0,
                "pages": set()
            }
        documents[chunk.filename]["total_chunks"] += 1
        documents[chunk.filename]["pages"].add(chunk.page_num)
    
    # Convert sets to counts
    doc_list = []
    for filename, data in documents.items():
        doc_list.append({
            "filename": filename,
            "total_chunks": data["total_chunks"],
            "total_pages": len(data["pages"])
        })
    
    return DatabaseStats(
        total_chunks=len(vector_db.chunks),
        total_documents=len(documents),
        documents=doc_list
    )

@app.delete("/clear")
async def clear_database():
    """Clear all uploaded documents and vector database"""
    try:
        # Clear vector database
        vector_db.clear()
        
        # Remove saved database files
        db_files = [f"{vector_db_path}.faiss", f"{vector_db_path}.json"]
        for db_file in db_files:
            if os.path.exists(db_file):
                os.remove(db_file)
        
        # Clear upload directory
        for file in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        return {"message": "Database and uploads cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

@app.get("/models")
async def get_available_models():
    """Get list of available Gemini models"""
    return {
        "available_models": [
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ],
        "default_model": "gemini-2.0-flash-exp",
        "description": "Gemini 2.0 Flash Experimental is recommended for best performance"
    }

if __name__ == "__main__":
    import uvicorn
    
    print("Starting PDF Q&A Backend with FastAPI, PyMuPDF, and Gemini 2.0 Flash...")
    print("Make sure to set your GEMINI_API_KEY in a .env file")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)