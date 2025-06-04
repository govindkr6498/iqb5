from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.responses import JSONResponse, FileResponse
import os
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import requests
from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from uuid import uuid4
from langchain_core.tools import tool
from datetime import datetime, timedelta
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session Management
class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.session_timeout = timedelta(hours=1)
    
    def create_session(self):
        session_id = str(uuid4())
        self.sessions[session_id] = {
            "chat_history": [],
            "created_at": datetime.now(),
            "last_accessed": datetime.now()
        }
        return session_id
    
    def get_session(self, session_id):
        if session_id in self.sessions:
            if datetime.now() - self.sessions[session_id]["last_accessed"] > self.session_timeout:
                del self.sessions[session_id]
                return None
            self.sessions[session_id]["last_accessed"] = datetime.now()
            return self.sessions[session_id]
        return None
    
    def cleanup_expired_sessions(self):
        now = datetime.now()
        expired = [sid for sid, session in self.sessions.items() 
                  if now - session["last_accessed"] > self.session_timeout]
        for sid in expired:
            del self.sessions[sid]

session_manager = SessionManager()

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SALESFORCE_CLIENT_SECRET = "67027AA5E4793A9FDCE0B13FA11E9FA2A41CA7C7270079D654B56EAC195DA91F"
SALESFORCE_AUTH_URL = "https://iqb4-dev-ed.develop.my.salesforce.com/services/oauth2/token"
SALESFORCE_CLIENT_ID = "3MVG9pRzvMkjMb6kXIMaUGyXNzwSMewmrdMKrZmsdv8ZJ1dRg9cockiUAcWLre745UP.WoR.vWMe0Gh8Q4x35"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env file!")
if not all([SALESFORCE_CLIENT_ID, SALESFORCE_CLIENT_SECRET, SALESFORCE_AUTH_URL]):
    raise ValueError("Salesforce credentials are missing from .env file!")

# Configuration
EXCEL_PATH = "C:/Users/admin/Documents/Document/Bot/src/FSTC_Contact_QA.xlsx"  
# C:/Users/admin/Documents/Document/Bot/src
vector_store = None
chat_history = []

# Data Models
class ChatRequest(BaseModel):
    message: str

class GraphState(TypedDict):
    question: str
    documents: List
    chat_history: List[BaseMessage]
    response: str
    user_info: Dict[str, str]

# Excel Processing
def process_excel(file_path=EXCEL_PATH):
    global vector_store
    
    try:
        df = pd.read_excel(file_path)
        # Convert DataFrame to natural language text
        text = ""
        for _, row in df.iterrows():
            description = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            text += f"{description}\n\n"
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(chunks, embeddings)
        print("Excel data processed successfully")
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        raise

# Initialize data processing
process_excel()

# Salesforce Integration
def create_salesforce_lead(lead_data):
    try:
        access_token, instance_url = get_salesforce_access_token()
        lead_endpoint = f"{instance_url}/services/data/v60.0/sobjects/Lead/"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        response = requests.post(lead_endpoint, json=lead_data, headers=headers, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error creating Salesforce lead: {str(e)}")
        return False

def get_salesforce_access_token():
    payload = {
        "grant_type": "client_credentials",
        "client_id": SALESFORCE_CLIENT_ID,
        "client_secret": SALESFORCE_CLIENT_SECRET,
    }
    response = requests.post(SALESFORCE_AUTH_URL, data=payload, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data["access_token"], data["instance_url"]

@tool
def callSalesforce(name: str, email: str,phone:str) -> str:
    """whenever a person is interested in call, schedule, visit, contact, call me, email me, 
        phone, interested,buy, purchase, meet, speak"""
    logger.info(f"user_info: {name}")
    return 'Your Name is {name} and email is {email} and phone number is {phone}'
tools = [callSalesforce]

# Workflow Functions
def retrieve_documents(state: GraphState):
    if not vector_store:
        return {"documents": [], "response": "Our property database isn't available right now. Please try again later."}
    
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        documents = retriever.invoke(state["question"])
        return {"documents": documents}
    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")
        return {"documents": [], "response": "I'm having trouble accessing the property information."}

def generate_response(state: GraphState):
    documents = state["documents"]
    user_input = state["question"]
    
    if not documents:
        return {"response": "I couldn't find information about that in our property listings."}

    context = "\n".join([doc.page_content for doc in documents])
    chat_history = state["chat_history"]

    # Build conversation history
    conv_history = "\n".join([
        f"User: {msg.content}" if isinstance(msg, HumanMessage) 
        else f"Assistant: {msg.content}" 
        for msg in chat_history[-4:]
    ])

    prompt = f"""You're a friendly real estate assistant. Answer questions naturally using the property information below.
`
Property Information:
{context}

Conversation History:
{conv_history}

User's Question: {user_input}

Guidelines:
- Be warm and professional
- Keep responses concise but helpful
- If asking for contact info, explain why politely
- Never make up information not in the data
- Don't mention you're looking at data

Response:"""
    try:
        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)
        model_with_tools = llm.bind_tools(tools) 
        response = model_with_tools.invoke(prompt)
        print('response:',response)
        print("Excel data processed successfully")
        return {"response": response.content}
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return {"response": "I'm having trouble formulating a response right now."}

def check_lead_generation(state: GraphState):
    user_input = state["question"].lower()
    triggers = [
        "schedule", "visit", "contact", "call me", "email me", 
        "phone", "interested", "buy", "purchase", "meet", "speak"
    ]
    return any(trigger in user_input for trigger in triggers)

def request_contact_info(state: GraphState):
    return {
        "response": "I'd be happy to connect you with our team! Could you please share your name, email, and phone number so we can follow up?"
    }

def extract_contact_info(state: GraphState):
    if "user_info" not in state:
        state["user_info"] = {}
    
    state["user_info"]["_attempts"] = state["user_info"].get("_attempts", 0) + 1
    
    if state["user_info"]["_attempts"] > 3:
        return {
            "response": "We'll have someone reach out to you soon. Thank you for your interest!",
            "user_info": state["user_info"]
        }
    
    user_input = state["question"]
    
    # Extract email
    if "email" not in state["user_info"]:
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_input)
        if email_match:
            state["user_info"]["email"] = email_match.group(0)
    
    # Extract phone
    if "phone" not in state["user_info"]:
        phone_match = re.search(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})', user_input)
        if phone_match:
            state["user_info"]["phone"] = re.sub(r'[^\d]', '', phone_match.group(0))[:15]
    
    # Extract name
    if "name" not in state["user_info"]:
        name_match = re.search(
            r'(?:name is|my name is|I\'m|I am|call me)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', 
            user_input, 
            re.IGNORECASE
        )
        if not name_match:
            name_match = re.search(r'([A-Z][a-z]+\s[A-Z][a-z]+)', user_input)
        if name_match:
            state["user_info"]["name"] = name_match.group(1).strip()
    
    missing = [field for field in ["name", "email", "phone"] if field not in state["user_info"]]
    
    if not missing:
        return {
            "response": f"Thank you {state['user_info']['name']}! We'll contact you soon at {state['user_info']['email']} or {state['user_info']['phone']}.",
            "user_info": state["user_info"]
        }
    else:
        prompts = {
            "name": "May I have your full name?",
            "email": "What email address should we use?",
            "phone": "And your phone number?"
        }
        return {
            "response": "Just a few more details: " + " ".join(prompts[field] for field in missing),
            "user_info": state["user_info"]
        }

# Workflow Construction
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate_response", generate_response)
workflow.add_node("request_contact", request_contact_info)
workflow.add_node("extract_contact", extract_contact_info)

workflow.set_entry_point("retrieve")

workflow.add_conditional_edges(
    "retrieve",
    check_lead_generation,
    {
        True: "request_contact",
        False: "generate_response"
    }
)

workflow.add_edge("request_contact", "extract_contact")

workflow.add_conditional_edges(
    "extract_contact",
    lambda state: all(k in state.get("user_info", {}) for k in ["name", "email", "phone"]) or 
                state.get("user_info", {}).get("_attempts", 0) > 3,
    {
        True: "generate_response",
        False: "request_contact"
    }
)

workflow.add_edge("generate_response", END)

app_graph = workflow.compile()

# API Endpoints
@app.post("/api/govind")
async def chat_endpoint(chat_request: ChatRequest, request: Request):
    try:
        # Session handling
        session_id = request.cookies.get("session_id")
        if not session_id or not session_manager.get_session(session_id):
            session_id = session_manager.create_session()
        
        session = session_manager.get_session(session_id)
        
        # Prepare initial state
        initial_state = {
            "question": chat_request.message,
            "documents": [],
            "chat_history": session["chat_history"],
            "response": "",
            "user_info": session.get("user_info", {})
        }
        
        # Execute workflow
        result = app_graph.invoke(initial_state)
        
        # Update session
        session["chat_history"].extend([
            HumanMessage(content=chat_request.message),
            AIMessage(content=result.get("response", ""))
        ])
        
        if "user_info" in result:
            session["user_info"] = result["user_info"]
        
        # Create Salesforce lead if complete info
        user_info = result.get("user_info", {})
        # print('user_info ',user_info)
        
        logger.info(f"user_info: {user_info}")
        if all(k in user_info for k in ["name", "email", "phone"]):
            lead_data = {
                "LastName": user_info["name"],
                "Phone": user_info["phone"],
                "Company": "Real Estate Inquiry",
                "Email": user_info["email"]
            }
            create_salesforce_lead(lead_data)
        
        # Prepare response
        response = JSONResponse({
            "answer": result.get("response", "I'm having trouble processing your request."),
            "user_info": user_info
        })
        
        # Set session cookie
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=3600,
            secure=True,
            samesite="lax"
        )
        
        return response
    
    except Exception as e:
        return JSONResponse(
            {"answer": "Sorry, I encountered an error processing your request.", "error": str(e)},
            status_code=500
        )
