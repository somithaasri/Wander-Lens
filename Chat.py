from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import os
import requests
import uuid
import PyPDF2
import io
import time
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import torch
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    "serpapi_key": os.getenv("SERPAPI_KEY"),
    "hf_token": os.getenv("HF_TOKEN"),
    "menu_dir": "menus"
}

class TripPlanner:
    def __init__(self):
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.llm = InferenceClient(token=CONFIG["hf_token"])
        self.articles = []

    def search_web(self, query: str):
        """Search Google using SerpAPI"""
        params = {
            "q": query,
            "api_key": os.getenv("SERPAPI_KEY"),
            "num": 10
        }
        response = requests.get("https://serpapi.com/search", params=params)
        return response.json().get("organic_results", [])

    def process_articles(self, results):
        """Extract and clean article content using BeautifulSoup"""
        articles = []
        for result in results:
            try:
                response = requests.get(result["link"], timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')

                # Remove unwanted elements
                for script in soup(["script", "style", "nav", "footer"]):
                    script.decompose()

                text = ' '.join([p.get_text() for p in soup.find_all('p')])
                articles.append({
                    "text": text[:2000],  # limit text length
                    "title": soup.title.string if soup.title else "",
                    "link": result["link"]
                })
            except Exception as e:
                print(f"Error processing {result['link']}: {str(e)}")
                continue
        return articles

    def store_in_memory(self, articles, location):
        """Store articles in memory"""
        for art in articles:
            self.articles.append({
                "location": location,
                "title": art["title"],
                "text": art["text"],
                "link": art["link"]
            })

    def generate_itinerary(self, user_input):
        """Main function to create travel plan"""
        # 1. Search web for information
        query = f"{user_input['trip_type']} activities near {user_input['location']} site:reddit.com OR site:tripadvisor.com"
        web_results = self.search_web(query)
        articles = self.process_articles(web_results)

        # 2. Store in memory
        self.store_in_memory(articles, user_input["location"])

        # 3. Find relevant information
        relevant_articles = [
            art for art in self.articles if art["location"] == user_input["location"]
        ]

        context = "\n".join([f"Source: {art['link']}\nContent: {art['title']}" for art in relevant_articles])

        prompt = f"""Create a {user_input['days']}-day itinerary for {user_input['location']}.
        - Hotel: {user_input['hotel']}
        - Budget: {user_input['budget']}
        - Trip Type: {user_input['trip_type']}

        Use these trusted sources:
        {context}

        Include hidden local gems and practical transportation info."""

        response = self.llm.text_generation(
            prompt,
            model="google/flan-t5-small",
            max_new_tokens=1000
        )

        return response

class ChatSession:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.history = []
        self.location = None
        self.last_activity = time.time()

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self.last_activity = time.time()

class ARMenuManager:
    def __init__(self):
        self.menus = self.load_existing_menus()

    def load_existing_menus(self):
        menus = {}
        if os.path.exists(CONFIG["menu_dir"]):
            for filename in os.listdir(CONFIG["menu_dir"]):
                if filename.endswith(".pdf"):
                    restaurant = filename[:-4]
                    menus[restaurant.lower()] = self.extract_menu_data(filename)
        return menus

    def extract_menu_data(self, filename: str):
        ar_link = None
        menu_items = []
        with open(os.path.join(CONFIG["menu_dir"], filename), "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if "AR_MODEL_LINK:" in text:
                    ar_link = text.split("AR_MODEL_LINK:")[1].split("\n")[0].strip()
                menu_items.append(text)
        return {
            "ar_link": ar_link,
            "full_menu": "\n".join(menu_items),
            "dishes": self.extract_dishes(menu_items)
        }

    def extract_dishes(self, menu_texts: List[str]):
        dishes = []
        for text in menu_texts:
            lines = text.split("\n")
            for line in lines:
                if "$" in line and len(line) < 100:
                    dishes.append(line.strip())
        return dishes

class TravelChatModel:
    def __init__(self):
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.llm = InferenceClient(token=CONFIG["hf_token"])
        self.sessions = {}
        self.menu_manager = ARMenuManager()

    def create_session(self):
        session = ChatSession()
        self.sessions[session.session_id] = session
        return session

    def get_local_reviews(self, query: str):
        params = {
            "q": f"{query} site:yelp.com OR site:tripadvisor.com",
            "api_key": CONFIG["serpapi_key"],
            "num": 5
        }
        response = requests.get("https://serpapi.com/search", params=params)
        return response.json().get("organic_results", [])

    def generate_response(self, session: ChatSession, user_message: str):
        # Check if asking about restaurants
        if "restaurant" in user_message.lower() or "eat" in user_message.lower():
            return self.handle_restaurant_query(session, user_message)

        # Check if asking about menu
        if "menu" in user_message.lower():
            return self.handle_menu_query(session, user_message)

        # General conversation
        prompt = self.build_conversation_prompt(session, user_message)

        response = self.llm.text_generation(
            prompt,
            model="mistralai/Mistral-7B-Instruct-v0.2",
            max_new_tokens=200
        )

        session.add_message("assistant", response)
        return response

    def handle_restaurant_query(self, session: ChatSession, query: str):
        # Get local reviews
        reviews = self.get_local_reviews(f"best {query} near {session.location}")
        formatted_reviews = "\n".join([
            f"{res['title']} ({res['link']}): {res['snippet']}"
            for res in reviews
        ])

        # Check PDF menus
        pdf_results = []
        for res in reviews:
            restaurant_name = res['title'].split(" - ")[0].lower()
            if restaurant_name in self.menu_manager.menus:
                menu_data = self.menu_manager.menus[restaurant_name]
                pdf_results.append({
                    "name": res['title'],
                    "dishes": menu_data['dishes'][:3],
                    "ar_link": menu_data['ar_link']
                })

        prompt = f"""User asked: {query}
        Location: {session.location}
        Found restaurants: {formatted_reviews}
        PDF menu data: {pdf_results}

        Provide a helpful response including:
        - Top 3 restaurant suggestions
        - Mention if we have AR menu preview available
        - Key highlights from reviews"""

        response = self.llm.text_generation(
            prompt,
            model="google/flan-t5-small",
            max_new_tokens=200
        )

        session.add_message("assistant", response)
        return response

    def build_conversation_prompt(self, session: ChatSession, new_message: str):
        history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in session.history[-5:]]
        )
        return f"""You are a travel assistant helping with {session.location}.
        Conversation History:
        {history}
        User: {new_message}
        Assistant:"""

    def extract_ar_link(self, response: str):
        if "AR_LINK:" in response:
            return response.split("AR_LINK:")[1].split(" ")[0].strip()
        return None

travel_chat_model = TravelChatModel()

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    ar_link: str = None

@app.post("/start_chat")
async def start_chat():
    session = travel_chat_model.create_session()
    return {"session_id": session.session_id}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    if not request.session_id or request.session_id not in travel_chat_model.sessions:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    session = travel_chat_model.sessions[request.session_id]
    session.add_message("user", request.message)

    if not session.location:
        # First message should set location
        session.location = request.message
        response = f"Welcome to GeoGuide! What would you like to know about {session.location}?"
    else:
        response = travel_chat_model.generate_response(session, request.message)

    ar_link = travel_chat_model.extract_ar_link(response)
    return ChatResponse(
        response=response,
        session_id=session.session_id,
        ar_link=ar_link if ar_link else ""
    )

# Unity AR Endpoint (Example)
@app.get("/ar_viewer/{model_id}")
async def serve_ar_model(model_id: str):
    return {"unity_build_url": f"https://your-ar-host.com/models/{model_id}"}

# Menu Upload Endpoint
@app.post("/upload_menu/{restaurant_name}")
async def upload_menu(restaurant_name: str, file: UploadFile = File(...)):
    file_content = await file.read()

    # Save PDF
    filename = f"{restaurant_name.replace(' ', '_')}.pdf"
    os.makedirs(CONFIG["menu_dir"], exist_ok=True)

    with open(os.path.join(CONFIG["menu_dir"], filename), "wb") as f:
        f.write(file_content)

    # Reload menus
    travel_chat_model.menu_manager.load_existing_menus()

    return {"status": "success", "filename": filename}

# Helper function to clean old sessions
async def cleanup_sessions():
    while True:
        now = time.time()
        to_remove = []
        for session_id, session in travel_chat_model.sessions.items():
            if now - session.last_activity > 3600:  # 1 hour timeout
                to_remove.append(session_id)
        for session_id in to_remove:
            del travel_chat_model.sessions[session_id]
        await asyncio.sleep(60)

# Start cleanup task
asyncio.create_task(cleanup_sessions())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
