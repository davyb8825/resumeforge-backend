from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from mcp.router import route_prompt
from utils.file_parser import parse_uploaded_file

# Load environment variables
load_dotenv()

app = FastAPI(title="ResumeForge API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "ResumeForge API is running!", "version": "1.0.0"}

@app.post("/prompt/")
async def prompt_handler(
    user_input: str = Form(...),
    expert: str = Form(...),
    job_title: str = Form(None),
    job_description: str = Form(None)
):
    """
    Main endpoint for processing user requests through MCP expert routing
    
    Args:
        user_input: Resume text or user query
        expert: Which expert to use (resume, cover_letter, interview)
        job_title: Optional job title for tailoring
        job_description: Optional job description for better context
    """
    try:
        result = await route_prompt(user_input, expert, job_title, job_description)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/upload/")
async def upload_resume(
    file: UploadFile = File(...),
    expert: str = Form(...),
    job_title: str = Form(None),
    job_description: str = Form(None)
):
    """
    Handle file uploads (PDF/DOCX) and process through MCP routing
    """
    try:
        # Parse the uploaded file
        file_content = await parse_uploaded_file(file)
        
        # Route to appropriate expert
        result = await route_prompt(file_content, expert, job_title, job_description)
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

@app.get("/experts/")
async def list_experts():
    """List available expert agents"""
    return {
        "experts": {
            "resume": {
                "name": "ResumeDoctor",
                "description": "Rewrites and enhances resumes for better impact"
            },
            "cover_letter": {
                "name": "CoverLetterWriter", 
                "description": "Creates job-specific cover letters"
            },
            "interview": {
                "name": "InterviewCoach",
                "description": "Generates tailored interview questions and prep"
            }
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    groq_key = os.getenv("GROQ_API_KEY")
    return {
        "status": "healthy",
        "groq_configured": bool(groq_key),
        "model": os.getenv("MODEL_NAME", "llama3-70b-8192")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
