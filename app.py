from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# API 키를 환경변수에서 가져옴 (Heroku config vars에서 설정한 값을 사용)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# FastAPI 앱 초기화
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
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4"
        )
        response = llm.invoke(messages)
        
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Heroku는 $PORT 환경변수를 제공합니다
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
