import httpx
import os
import json
from typing import List, Dict, Optional
from mcp.planner import resume_doctor
from mcp.cover_letter import generate_cover_letter
from mcp.interview import interview_questions

class GroqClient:
    """GROQ API client for LLM interactions"""
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("MODEL_NAME", "llama3-70b-8192")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
    
    async def query_groq(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Send messages to GROQ API and return response
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Response creativity (0.0-1.0)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4000,
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.post(self.base_url, headers=headers, json=payload)
                response.raise_for_status()
                
                data = response.json()
                return data["choices"][0]["message"]["content"]
                
            except httpx.HTTPStatusError as e:
                raise Exception(f"GROQ API error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                raise Exception(f"Request failed: {str(e)}")

# Initialize GROQ client
groq_client = GroqClient()

async def route_prompt(
    user_input: str, 
    expert: str, 
    job_title: Optional[str] = None,
    job_description: Optional[str] = None
) -> Dict:
    """
    Route user input to appropriate MCP expert
    
    Args:
        user_input: Resume text or user query
        expert: Expert type (resume, cover_letter, interview)
        job_title: Optional job title for context
        job_description: Optional job description for better tailoring
    """
    try:
        # Route to appropriate expert with GROQ query function
        if expert == "resume":
            return await resume_doctor(user_input, groq_client.query_groq, job_title, job_description)
            
        elif expert == "cover_letter":
            if not job_title:
                return {"error": "Job title is required for cover letter generation"}
            return await generate_cover_letter(user_input, job_title, groq_client.query_groq, job_description)
            
        elif expert == "interview":
            return await interview_questions(user_input, groq_client.query_groq, job_title, job_description)
            
        else:
            return {"error": f"Unknown expert: {expert}. Available: resume, cover_letter, interview"}
            
    except Exception as e:
        return {"error": f"Expert routing failed: {str(e)}"}

async def test_groq_connection() -> Dict:
    """Test GROQ API connection"""
    try:
        test_messages = [{"role": "user", "content": "Say 'GROQ connection successful'"}]
        response = await groq_client.query_groq(test_messages)
        return {"status": "success", "response": response}
    except Exception as e:
        return {"status": "error", "message": str(e)}
