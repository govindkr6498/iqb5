from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
from sales_rag_bot import SalesRAGBot
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Sales RAG Bot API",
    description="An API for interacting with the Sales RAG chatbot",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the chatbot
pdf_path = '/home/ubuntu/iqb5/PropertyDetail.pdf'
chatbot = SalesRAGBot(pdf_path)

class ChatInput(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    lead_info: Optional[Dict[str, Any]] = None
    lead_state: str

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_input: ChatInput):
    """
    Process a chat message and return the bot's response
    """
    try:
        result = chatbot.process_message(chat_input.message)
        return ChatResponse(
            response=result['response'],
            lead_info=result['lead_info'],
            lead_state=result['lead_state']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     """
#     Root endpoint to verify the API is running
#     """
#     return {
#         "status": "ok",
#         "message": "Sales RAG Bot API is running"
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
