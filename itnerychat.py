import os
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub import split_torch_state_dict_into_shards
import torch
import time

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    "serpapi_key": os.getenv("SERPAPI_KEY"),
    "hf_token": os.getenv("HF_TOKEN")
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
            model="mistralai/Mistral-7B-Instruct-v0.2",
            max_new_tokens=1000
        )
        
        return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
