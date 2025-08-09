
import io
from fastapi import UploadFile, HTTPException

async def parse_uploaded_file(file: UploadFile) -> str:
    """Parse uploaded files and extract text"""
    
    # Check file size (10MB limit)
    file_content = await file.read()
    if len(file_content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    
    try:
        if file.filename.lower().endswith('.txt'):
            return file_content.decode('utf-8')
        elif file.filename.lower().endswith('.pdf'):
            return await parse_pdf_basic(file_content)
        else:
            raise HTTPException(
                status_code=400, 
                detail="Supported formats: TXT, PDF only"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File parsing error: {str(e)}")

async def parse_pdf_basic(file_content: bytes) -> str:
    """Basic PDF parsing with PyPDF2"""
    try:
        import PyPDF2
        pdf_file = io.BytesIO(file_content)
        reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        if not text.strip():
            raise Exception("No text extracted from PDF. Try a different PDF or convert to TXT.")
        
        return clean_text(text)
    except Exception as e:
        raise Exception(f"PDF parsing failed: {str(e)}")

def clean_text(text: str) -> str:
    """Clean extracted text"""
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(cleaned_lines)
