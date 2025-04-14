from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path
import re
from fastapi.responses import JSONResponse
import requests
from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from uuid import uuid4
from fastapi import Request
from datetime import datetime, timedelta

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.mount("/static", StaticFiles(directory="static"), name="static")

class SessionManager:
    def __init__(self):
        self.sessions = {}  # Stores session_id: {data}
        self.session_timeout = timedelta(hours=1)  # Sessions expire after 1 hour
    
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
            # Check if session expired
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

# Initialize session manager
session_manager = SessionManager()

@app.get("/")
async def root():
    return {"message": "FastAPI is running!"}

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SALESFORCE_CLIENT_SECRET = "67027AA5E4793A9FDCE0B13FA11E9FA2A41CA7C7270079D654B56EAC195DA91F"
SALESFORCE_AUTH_URL = "https://iqb4-dev-ed.develop.my.salesforce.com/services/oauth2/token"
SALESFORCE_CLIENT_ID = "3MVG9pRzvMkjMb6kXIMaUGyXNzwSMewmrdMKrZmsdv8ZJ1dRg9cockiUAcWLre745UP.WoR.vWMe0Gh8Q4x35"

if not OPENAI_API_KEY:
    raise ValueError("ERROR: OPENAI_API_KEY is missing from .env file!")

if not SALESFORCE_CLIENT_ID or not SALESFORCE_CLIENT_SECRET or not SALESFORCE_AUTH_URL:
    raise ValueError("ERROR: Salesforce credentials are missing from .env file!")


# PDF_PATH = "C:/Users/lenovo/Documents/Boat/src/PropertyDetail.pdf"
PDF_PATH = "/home/ubuntu/iqb5/PropertyDetail.pdf"

vector_store = None
chat_history = []


class ChatRequest(BaseModel):
    message: str

class GraphState(TypedDict):
    question: str
    documents: List
    chat_history: List[BaseMessage]
    response: str
    user_info: Dict[str, str]

def process_pdf(file_path=PDF_PATH):
    global vector_store
    pdf_reader = PdfReader(file_path)
    text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])

    # Split text into smaller chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Convert text chunks into embeddings
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)

process_pdf(PDF_PATH)



def create_salesforce_lead(lead_data):
    """Creates a new Salesforce lead using provided contact details."""
    access_token, instance_url = get_salesforce_access_token()
    lead_endpoint = f"{instance_url}/services/data/v60.0/sobjects/Lead/" 
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(lead_endpoint, json=lead_data, headers=headers, timeout=10)
    print('n  :',response)
    if response.status_code == 201:
        print(f"Thank you {lead_data['LastName']}! We'll contact you at {lead_data['Email']} or {lead_data['Phone']}.")
        # return {
        #     "response": f"Thank you {lead_data['LastName']}! We'll contact you at {lead_data['Email']} or {lead_data['Phone']}.",
        #     "user_info": lead_data
        # }
    elif response.status_code == 400 and "DUPLICATES_DETECTED" in response.text:
        print('Successfully Created')
        # return {
        #     "response": "A meeting has already been scheduled for this user. Our team will be in touch with you shortly.",
        #     "user_info": lead_data
        # }
    else:
        print('fail Created')
        # return {
        #     "response": f"Something went wrong while creating your meeting: {response.text}",
        #     "user_info": lead_data
        # }

def get_salesforce_access_token():
    """Fetches a Salesforce access token dynamically."""
    payload = {
        "grant_type": "client_credentials",
        "client_id": SALESFORCE_CLIENT_ID,
        "client_secret": SALESFORCE_CLIENT_SECRET,
    }

    response = requests.post(SALESFORCE_AUTH_URL, params=payload, timeout=10)
    if response.status_code == 200:
        data = response.json()
        return data["access_token"], data["instance_url"]
    else:
        raise ValueError(f"Salesforce Auth Failed: {response.text}")
    
    
def retrieve_documents(state: GraphState):
    global vector_store
    if not vector_store:
        return {"documents": [], "response": "No document uploaded yet. Please upload a document first."}
    
    user_input = state["question"]
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    documents = retriever.invoke(user_input)
    return {"documents": documents}

def generate_response(state: GraphState):                
    # global chat_history
    documents = state["documents"]
    user_input = state["question"]
    chat_history = state["chat_history"]
    if not documents:
        return {"response": "I don't know. The document does not contain relevant information."}

    context = "\n".join([doc.page_content for doc in documents])

    # Add user's message to chat history
    chat_history.append(HumanMessage(content=user_input))

    if len(chat_history) > 5:
        chat_history = chat_history[-5:]

    # Create the conversation prompt for the LLM
    conversation_history = "\n".join([message.content for message in chat_history])

    prompt = (f"You are an interactive AI sales agent assisting potential real estate buyers. "
        f"Your job is to answer queries strictly based on the provided document and conversation history. "
        f"You should also engage users in a natural, conversational manner while ensuring their questions are answered effectively.\n\n"
        
        f"### Instructions for Response:\n"
        f"- If the user requests property details, return data in *markdown table format*.\n"
        f"- **If the user expresses interest in purchasing, visiting, or speaking to a sales agent**, collect their **Name, Email, and Phone Number** politely and confirm their intent.\n"
        f"- **If the user asks for information outside the document**, respond with 'I don't know' and guide them back to relevant topics.\n"
        f"- **Maintain a friendly and professional tone**, and ensure smooth, natural interactions.\n\n"
)


     # Add user contact info if it exists
    if state.get("user_info"):
        prompt += (
            f"\n**User Contact Information:**\n"
            f"Name: {state['user_info'].get('name', 'Not provided')}\n"
            f"Email: {state['user_info'].get('email', 'Not provided')}\n"
            f"Phone: {state['user_info'].get('phone', 'Not provided')}\n\n"
        )

    # Append the remaining parts of the prompt
    prompt += (
        f"### Conversation Context:\n"
        f"**Document Context:**\n{context}\n\n"
        f"**Conversation History:**\n{conversation_history}\n\n"
        f"**User's Question:** {user_input}\n\n"
        f"Based on the user's question, respond appropriately while following these instructions."
    )    

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    response = llm.invoke(prompt)
    
    # Add bot's response to chat history
    bot_response = response.content
    chat_history.append(AIMessage(content=bot_response))
    
    return {"response": bot_response}

def check_lead_generation(state: GraphState):
    """Check if the user wants to provide contact info or schedule a visit"""
    user_input = state["question"].lower()
    triggers = ["schedule", "visit", "contact", "call me", "email me", "phone", "interested", "buy", "purchase"]
    return any(trigger in user_input for trigger in triggers)

def request_contact_info(state: GraphState):
    """Ask for contact information when user shows interest"""
    return {"response": "I'd be happy to connect you with our sales team! Could you please provide your name, email, and phone number so we can follow up with you?"}

def extract_contact_info(state: GraphState):
    """Improved contact info extraction with recursion protection"""
    # Initialize user_info if not present
    if "user_info" not in state:
        state["user_info"] = {}
    
    # Track how many times we've tried to extract info
    state["user_info"]["_attempts"] = state["user_info"].get("_attempts", 0) + 1
    
    # If we've tried too many times, give up
    if state["user_info"]["_attempts"] > 3:
        return {
            "response": "We'll connect you with our sales team shortly. Thank you for your interest!",
            "user_info": state["user_info"]
        }
    
    user_input = state["question"]
    
    
    if "email" not in state["user_info"]:
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_input)
        if email_match:
            state["user_info"]["email"] = email_match.group(0)
    
    if "phone" not in state["user_info"]:
        phone_match = re.search(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})', user_input)
        if phone_match:
            state["user_info"]["phone"] = re.sub(r'[^\d]', '', phone_match.group(0))[:15]
    
    # if "name" not in state["user_info"]:
    #     name_match = re.search(r'([A-Za-z]{2,}(?:\s[A-Za-z]{2,})+)', user_input)
    #     if name_match:
    #         state["user_info"]["name"] = name_match.group(0).strip()
    if "name" not in state["user_info"]:
        # Case 1: User says "My name is John Doe"
        name_match = re.search(
            r'(?:name is|my name is|I\'m|I am|call me)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', 
            user_input, 
            re.IGNORECASE
        )
        
        # Case 2: Fallback - look for capitalized words (e.g., "John Doe")
        if not name_match:
            name_match = re.search(r'([A-Z][a-z]+\s[A-Z][a-z]+)', user_input)
        
        if name_match:
            state["user_info"]["name"] = name_match.group(1).strip()
    
    missing = [
        field for field in ["name", "email", "phone"] 
        if field not in state["user_info"]
    ]
    
    if not missing:
        # lead_data={
        #     "LastName":state['user_info']['name'],
        #     "Phone":state['user_info']['phone'],
        #     "Company":"Iquestbee Technology",
        #     "Email":state['user_info']['email']
        # }
        # return create_salesforce_lead(lead_data)
        return {
            "response": f"Thank you {state['user_info']['name']}! We'll contact you at {state['user_info']['phone']} or {state['user_info']['email']}.",
            "user_info": state["user_info"]
        }
    else:
        # Ask specifically for what's missing
        prompts = {
            "name": "May I have your full name?",
            "email": "Could you please provide your email address?",
            "phone": "What's the best phone number to reach you?"
        }
        return {
            "response": " ".join(prompts[field] for field in missing),
            "user_info": state["user_info"]
        }
def check_contact_info_provided(state: GraphState):
    """Check if all contact info has been provided"""
    user_info = state.get("user_info", {})
    return all(key in user_info for key in ["name", "email", "phone"])

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
        True: "generate_response",  # Got all info or gave up
        False: "request_contact"   # Go back to ask again (but with updated state)
    }
)

workflow.add_edge("generate_response", END)

app_graph = workflow.compile()

@app.post("/api/govind")
async def ask_question(chat_request: ChatRequest, request: Request):
    try:
        # Get or create session
        session_id = request.cookies.get("session_id")
        if not session_id or not session_manager.get_session(session_id):
            session_id = session_manager.create_session()
        
        session = session_manager.get_session(session_id)
        
        initial_state = {
            "question": chat_request.message,
            "documents": [],
            "chat_history": session["chat_history"],
            "response": "",
            "user_info": session.get("user_info", {})  # Start with empty user_info
        }
        
        result = app_graph.invoke(initial_state)
        
        # Update session data
        if "chat_history" in result:
            session["chat_history"] = result["chat_history"]
    
        # Store user info if provided
        if "user_info" in result:
            session["user_info"] = result["user_info"]
        
        response_data = {
            "answer": result.get("response", "Sorry, I encountered an error."),
            "user_info": result.get("user_info", {})
        }
        
        # If we have complete info, log it (in real app, store in DB)
        if all(k in response_data["user_info"] for k in ["name", "email", "phone"]):
            print(f"✅ Lead captured: {response_data['user_info']}")
            
            lead_data = {
                "LastName": response_data['user_info']['name'],
                "Phone": response_data['user_info']['phone'],
                "Company": "Iquestbee Technology",
                "Email": response_data['user_info']['email']
            }
            create_salesforce_lead(lead_data)
        
        # Create response with session cookie       
        response = JSONResponse(response_data)
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=3600,  # 1 hour
            secure=True,  # In production, set this to True
            samesite="lax"
        )
        return response
    
    except Exception as e:
        return {"answer": f"Sorry, I encountered an error: {str(e)}", "user_info": {}}
    


@app.get("/", response_class=FileResponse)
async def serve_index(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or not session_manager.get_session(session_id):
        session_id = session_manager.create_session()
    
    response = FileResponse("C:/Users/lenovo/Documents/Boat/src/index.html")
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        max_age=3600,  # 1 hour
        secure=True,  # In production, set this to True
        samesite="lax"
    )
    return response
