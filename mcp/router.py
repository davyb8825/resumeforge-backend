import httpx
import os
from typing import List, Dict, Optional

class GroqClient:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = "llama3-70b-8192"
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found")
    
    async def query_groq(self, messages, temperature=0.7):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4000
        }
        
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.post(self.base_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                raise Exception(f"GROQ API error: {str(e)}")

# Initialize client
groq_client = GroqClient()

async def resume_doctor(resume_text, query_fn, job_title=None, job_description=None):
    context = ""
    if job_title:
        context += f"\nTarget Job Title: {job_title}"
    if job_description:
        context += f"\nJob Requirements: {job_description[:500]}..."
    
    prompt = f"""You are ResumeDoctor, an expert resume writer. 

{context}

ORIGINAL RESUME:
{resume_text}

Please rewrite this resume to:
- Improve clarity and structure
- Use strong action verbs
- Add quantifiable achievements where possible
- Make it ATS-friendly
- Enhance weak descriptions

Return only the enhanced resume content."""
    
    try:
        result = await query_fn([{"role": "user", "content": prompt}])
        return {"expert": "ResumeDoctor", "result": result, "status": "success"}
    except Exception as e:
        return {"expert": "ResumeDoctor", "error": str(e), "status": "error"}

async def generate_cover_letter(resume_text, job_title, query_fn, job_description=None):
    context = f"Job Title: {job_title}"
    if job_description:
        context += f"\n\nJob Description:\n{job_description}"
    
    prompt = f"""You are CoverLetterWriter, an expert in crafting compelling cover letters.

{context}

CANDIDATE'S RESUME:
{resume_text}

Create a professional cover letter that:
- Connects the candidate's experience to the job requirements
- Shows genuine interest in the role
- Highlights key achievements
- Has a strong opening and closing
- Is concise but impactful

Return only the cover letter content."""
    
    try:
        result = await query_fn([{"role": "user", "content": prompt}])
        return {"expert": "CoverLetterWriter", "result": result, "status": "success"}
    except Exception as e:
        return {"expert": "CoverLetterWriter", "error": str(e), "status": "error"}

async def interview_questions(resume_text, query_fn, job_title=None, job_description=None):
    context = ""
    if job_title:
        context += f"Target Role: {job_title}\n"
    if job_description:
        context += f"Job Requirements: {job_description[:400]}...\n"
    
    prompt = f"""You are InterviewCoach, an expert interview preparation specialist.

{context}

CANDIDATE'S BACKGROUND:
{resume_text}

Generate comprehensive interview preparation with:

1. LIKELY QUESTIONS (8-10 questions they'll probably ask)
2. SAMPLE ANSWERS for 3-4 key questions using STAR method
3. QUESTIONS TO ASK THEM (5-6 thoughtful questions)
4. KEY TALKING POINTS (3-4 main strengths to highlight)

Make it specific to this candidate and role."""
    
    try:
        result = await query_fn([{"role": "user", "content": prompt}])
        return {"expert": "InterviewCoach", "result": result, "status": "success"}
    except Exception as e:
        return {"expert": "InterviewCoach", "error": str(e), "status": "error"}

async def route_prompt(user_input, expert, job_title=None, job_description=None):
    try:
        if expert == "resume":
            return await resume_doctor(user_input, groq_client.query_groq, job_title, job_description)
        elif expert == "cover_letter":
            if not job_title:
                return {"error": "Job title required for cover letter"}
            return await generate_cover_letter(user_input, job_title, groq_client.query_groq, job_description)
        elif expert == "interview":
            return await interview_questions(user_input, groq_client.query_groq, job_title, job_description)
        else:
            return {"error": f"Unknown expert: {expert}"}
    except Exception as e:
        return {"error": f"Routing failed: {str(e)}"}
