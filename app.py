import os
import uuid
import logging
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles 
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn
from config import config, logger
from agents.simple_chat_agent import chat_agent
from agents.rag_agent import rag_agent
from agents.multi_agent_orchestrator import multi_agent_orchestrator
from agents.voice_processor.voice_agent import VoiceAgent
from agents.rag_agent import vector_store, init_global_vector_store, close_global_vector_store

# Track app start time
app_start_time = datetime.now()

# Lifespan context for startup/shutdown (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Application starting up...")
        # Initialize global vector store lazily to avoid double-opening in reload workers
        initialized = init_global_vector_store()
        if initialized:
            logger.info("Vector store initialized (singleton)")
        else:
            logger.info("Vector store already initialized")
        yield
    finally:
        try:
            logger.info("Application shutting down...")
            if close_global_vector_store():
                logger.info("Vector store connection closed")
            else:
                logger.info("Vector store was not initialized or already closed")
        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")

# Initialize FastAPI app
app = FastAPI(
    title="Medical Assistant - Part 1",
    description="Basic FastAPI setup with file upload and chat endpoints",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Voice agent singleton for API endpoints
voice_agent = VoiceAgent()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up template (for later parts)
templates = Jinja2Templates(directory="templates")

# Add static files mounting for CSS, JS, and other assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error on {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": exc.errors()}
    )

# Pydantic models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="Chat message")

class ChatResponse(BaseModel):
    status: str
    response: str
    timestamp: str
    request_id: str

class UploadResponse(BaseModel):
    status: str
    message: str
    file_path: str
    file_size: int
    timestamp: str
    request_id: str

class FileInfo(BaseModel):
    filename: str
    size: int
    created: str
    extension: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime: str

# Routes
@app.get("/", response_class=HTMLResponse)  
async def read_root(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = str(datetime.now() - app_start_time)
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime=uptime
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, searchType: str = None):
    """Handle text chat messages with LLM integration""" 
    request_id = str(uuid.uuid4())
    logger.info(f"Chat request received: {request_id}")

    try:
        if config.ENABLE_AGENT_ROUTING:
            # Propagate optional search type preference to orchestrator via AgentState.messages meta
            # For simplicity, overload conversation history param to carry a tiny meta message
            history = chat_agent.get_history() or []
            if searchType:
                history.append({"role": "system", "content": f"SEARCH_TYPE={searchType}"})
            result = multi_agent_orchestrator.process_query(request.message, vector_store, history)
            logger.info(f"Multi-agent response generated: {request_id}, agent: {result['agent']}")
        else:
            #Fallback to simple RAG + chat system
            logger.info("Falling back to simple RAG + chat system")
            rag_result = rag_agent.process_query(request.message)
            
            if rag_result["confidence"] >= config.RAG_CONFIDENCE_THRESHOLD:
                # If RAG has good confidence, use it
                result = rag_result
            else:
                # Fall back to simple chat agent
                result = chat_agent.process_message(request.message)

        return ChatResponse(
            status=result["status"],
            response=result["response"],
            timestamp=result["timestamp"],
            request_id=request_id
        )
    except Exception as e:
        logger.error(f"Chat error {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ----- Voice I/O endpoints -----
@app.post("/voice/stt")
async def voice_to_text(audio_base64: str = Form(...)):
    """Convert base64-encoded audio to text using Azure STT."""
    try:
        result = voice_agent.process_voice_input(audio_base64)
        if result.get("status") != "success":
            raise HTTPException(status_code=400, detail=result.get("error", "STT failed"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/tts")
async def text_to_voice(text: str = Form(...), voice_id: str = Form("") ):
    """Convert text to base64 audio (data URL) using Azure TTS."""
    try:
        voice_name = voice_id or None
        result = voice_agent.generate_voice_response(text, voice_name)
        if result.get("status") != "success":
            raise HTTPException(status_code=400, detail=result.get("error", "TTS failed"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add new multi-agent endpoints
@app.get("/agents/info")
async def get_agents_info():
    """Get information about available agents"""
    try:
        return {
            "status": "success",
            "agents": [
                {
                    "name": "CONVERSATION_AGENT",
                    "description": "Handles general chat, greetings, and non-medical questions",
                    "status": "active"
                },
                {
                    "name": "RAG_AGENT",
                    "description": "Retrieves medical knowledge from document database",
                    "status": "active"
                },
                {
                    "name": "WEB_SEARCH_AGENT",
                    "description": "Fetches real-time medical information from web sources",
                    "status": "placeholder"
                },
                {
                    "name": "IMAGE_ANALYSIS_AGENT",
                    "description": "Analyzes medical images (brain MRI, chest X-ray, skin lesions) - routes to specialized medical agents",
                    "status": "active"
                },
                {
                    "name": "BRAIN_TUMOR_AGENT",
                    "description": "Analyzes brain MRI images for tumor detection and brain conditions",
                    "status": "active"
                },
                {
                    "name": "CHEST_XRAY_AGENT",
                    "description": "Analyzes chest X-ray images for lung conditions and pneumonia",
                    "status": "active"
                },
                {
                    "name": "SKIN_LESION_AGENT",
                    "description": "Analyzes skin lesion images for moles, rashes, and skin conditions",
                    "status": "active"
                }
            ],
            "routing_enabled": config.ENABLE_AGENT_ROUTING,
            "confidence_threshold": config.AGENT_CONFIDENCE_THRESHOLD,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting agents info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/route")
async def route_query(request: ChatRequest):
    """Manually route a query to test agent decision"""
    try:
        decision = multi_agent_orchestrator.agent_decision_system.decide_agent(request.message)
        return {
            "status": "success",
            "query": request.message,
            "decision": decision,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error routing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image-analysis/info")
async def get_image_analysis_info():
    """Get information about available medical image analysis agents"""
    try:
        return {
            "status": "success",
            "agents": [
                {
                    "name": "BRAIN_TUMOR_AGENT",
                    "description": "Analyzes brain MRI images for tumor detection and brain conditions",
                    "image_types": ["brain_mri", "brain", "mri", "tumor", "neurology"],
                    "status": "active"
                },
                {
                    "name": "CHEST_XRAY_AGENT", 
                    "description": "Analyzes chest X-ray images for lung conditions and pneumonia",
                    "image_types": ["chest_xray", "chest", "xray", "x-ray", "lung", "pneumonia"],
                    "status": "active"
                },
                {
                    "name": "SKIN_LESION_AGENT",
                    "description": "Analyzes skin lesion images for moles, rashes, and skin conditions",
                    "image_types": ["skin_lesion", "skin", "lesion", "mole", "rash", "dermatology"],
                    "status": "active"
                }
            ],
            "routing_strategy": "multimodal_llm_based",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting image analysis info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/chat/history')
async def get_chat_history():
    """Get conversation history summary"""
    try:
        summary = chat_agent.get_history_summary()
        return {
            "status": "success",
            "history": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/chat/clear')
async def clear_chat_history():
    """Clear conversation history"""
    try:
        chat_agent.clear_history()
        return {
            "status": "success",
            "message": "Conversation history cleared",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add new RAG-specific endpoints
@app.post("/rag/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a document into the knowledge base"""
    try:
        
        file_path = f"temp_{file.filename}"
        with open(file_path, 'wb') as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Ingest the document
        result = rag_agent.ingest_document(file_path)

        # Clean up temporary file
        os.remove(file_path)

        return result

    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/info")
async def get_rag_info():
    """Get information about the RAG knowledge base"""
    try:
        return rag_agent.get_knowledge_base_info()
    except Exception as e:
        logger.error(f"Error getting RAG info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/clear")
async def clear_knowledge_base():
    """Clear the knowledge base"""
    try:
        success = vector_store.clear_collection()
        if success:
            return {
                "status": "success",
                "message": "Knowledge base cleared successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to clear knowledge base")
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    image: UploadFile = File(..., description="The image file to upload"),
    text: str = Form("", description="Optional text to associate with the file")
):
    """Handle file uploads with optional text""" 
    request_id = str(uuid.uuid4())
    logger.info(f"Upload request received: {request_id}, file: {image.filename if image else 'None'}, text: {text}")
    
    # Check if file is provided
    if not image or not image.filename:
        logger.error(f"No file provided in upload request: {request_id}")
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )
    
    try:
        # Validate file type
        if not config.is_allowed_file(image.filename):
            logger.warning(f"File type not allowed: {image.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {config.ALLOWED_EXTENSIONS}"
            )
        
        # Read file content
        content = await image.read()

        # check file size
        if len(content) > config.MAX_FILE_SIZE: 
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {config.get_max_file_size_mb():.1f}MB"
            )

        # Generate unique filename
        file_extension = image.filename.split('.')[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        
        frontend_upload_dir = os.path.join(config.UPLOAD_FOLDER, "frontend")
        os.makedirs(frontend_upload_dir, exist_ok=True)
        
        file_path = os.path.join(frontend_upload_dir, unique_filename)

        # Save file
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Prepare response
        response_message = f"File '{image.filename}' uploaded successfully"
        if text and text.strip():
            response_message += f" with text: '{text}'"

        logger.info(f"File uploaded successfully: {request_id}, path: {file_path}")

        return UploadResponse(
            status="success",
            message=response_message,
            file_path=file_path,
            file_size=len(content),
            timestamp=datetime.now().isoformat(),
            request_id=request_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint for image analysis
@app.post("/analyze-image", response_model=ChatResponse)
async def analyze_image(
    image: UploadFile = File(..., description="The medical image to analyze"),
    query: str = Form("", description="Optional question about the image")
):
    """Analyze a medical image using the appropriate specialized medical agent (NO RAG)"""
    request_id = str(uuid.uuid4())
    logger.info(f"Image analysis request received: {request_id}, file: {image.filename if image else 'None'}, query: '{query}'")
    
    # Validate inputs
    if not image or not image.filename:
        logger.error(f"No image file provided in analyze request: {request_id}")
        raise HTTPException(status_code=400, detail="No image file provided")
    
    try:
        # Validate file type
        if not config.is_allowed_file(image.filename):
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {config.ALLOWED_EXTENSIONS}"
            )
        
        # Read and save file temporarily
        content = await image.read()
        file_extension = image.filename.split('.')[-1]
        temp_filename = f"temp_{uuid.uuid4()}.{file_extension}"
        temp_path = os.path.join(config.UPLOAD_FOLDER, temp_filename)
        
        with open(temp_path, "wb") as buffer:
            buffer.write(content)
        
        try:
            # Process the image through the multi-agent system (NO RAG)
            if config.ENABLE_AGENT_ROUTING:
                result = multi_agent_orchestrator.process_image_query(
                    query or "Please analyze this medical image",
                    temp_path,
                    vector_store
                )
            else:
                # Fallback to simple response
                result = {
                    "status": "success",
                    "response": f"Image '{image.filename}' uploaded successfully. Image analysis requires agent routing to be enabled.",
                    "agent": "UPLOAD_ONLY",
                    "confidence": 0.5,
                    "timestamp": datetime.now().isoformat()
                }
            
            return ChatResponse(
                status=result["status"],
                response=result["response"],
                timestamp=result["timestamp"],
                request_id=request_id
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis error {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files", response_model=dict)
async def list_files():
    """List uploaded files"""
    try:
        files = []
        total_size = 0

        for filename in os.listdir(config.UPLOAD_FOLDER):
            file_path = os.path.join(config.UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                file_info = FileInfo(
                    filename=filename,
                    size=stat.st_size,
                    created=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    extension=filename.split('.')[-1] if '.' in filename else 'unknown'
                )
                files.append(file_info)
                total_size += stat.st_size
        
        logger.info(f"Files listed: {len(files)} files, {total_size} bytes")
        
        return {
            "status": "success",
            "files": files,
            "count": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }

    except Exception as e:
        logger.error(f"List files error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete a specific file"""
    try:
        file_path = os.path.join(config.UPLOAD_FOLDER, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        os.remove(file_path)
        logger.info(f"File deleted: {filename}")
        
        return {
            "status": "success",
            "message": f"File '{filename}' deleted successfully",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete file error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    print(f"Starting Medical Assistant API on {config.HOST}:{config.PORT}")
    print(f"Debug mode: {config.DEBUG}")
    print(f"Upload folder: {config.UPLOAD_FOLDER}")
    logger.info(f"Max file size: {config.get_max_file_size_mb():.1f}MB")
    logger.info(f"Allowed extensions: {config.ALLOWED_EXTENSIONS}")

    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )