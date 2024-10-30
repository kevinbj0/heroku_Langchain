from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# API 키를 환경변수에서 가져옴
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="Story Generator API",
    description="An API that generates stories using OpenAI's GPT-4",
    version="1.0.0"
)

# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages([
    ("system", "Tell me a long story about {topic}")
])

class ChatInput(BaseModel):
    topic: str

class ChatOutput(BaseModel):
    response: str

@app.get("/")
async def root():
    return {"message": "Welcome to Story Generator API. Use /openai/ endpoint to generate stories."}

@app.post("/openai/", response_model=ChatOutput)
async def openai_endpoint(chat_input: ChatInput):
    try:
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
            
        messages = prompt.format_messages(topic=chat_input.topic)
        llm = ChatOpenAI(model="gpt-4")
        response = llm.invoke(messages)
        
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)