from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import os
import requests
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import googleapiclient.discovery

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

CONFIG = {
    "serpapi_key": os.getenv("SERPAPI_KEY"),
    "hf_token": os.getenv("HF_TOKEN"),
    "youtube_api_key": os.getenv("YOUTUBE_API_KEY")
}

class TripPlanner:
    def __init__(self):
        self.llm = InferenceClient(token=CONFIG["hf_token"])
        self.youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=CONFIG["youtube_api_key"])
        
    def get_vr_links(self, places: List[str]) -> Dict[str, str]:
        """Get YouTube videos for places"""
        vr_links = {}
        for place in places:
            try:
                request = self.youtube.search().list(
                    q= f"{place} 360",
                    part="snippet",
                    type="video",
                    maxResults=1
                )
                response = request.execute()
                
                videos = []
                for item in response.get("items", []):
                    video_id = item["id"]["videoId"]
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    videos.append(video_url)
                
                if videos:
                    vr_links[place] = videos[0]
                else:
                    vr_links[place] = "No link found"
            except Exception as e:
                print(f"VR search error for {place}: {str(e)}")
            finally:
                print(f"VR links for {place}: {vr_links.get(place, 'No link found')}")
        return vr_links

    def generate_itinerary(self, user_input):
        # Generate base itinerary
        prompt = f"""Create a {user_input['days']}-day itinerary for {user_input['location']}.
        - Hotel: {user_input['hotel']}
        - Budget: {user_input['budget']}
        - Trip Type: {user_input['trip_type']}
        
        Include hidden local gems and practical transportation info.
        Format: Put each place name on a new line starting with '- '"""
        
        response_text = self.llm.text_generation(
            prompt,
            model="mistralai/Mistral-7B-Instruct-v0.2",
            max_new_tokens=1500
        )
        
        try:
            # Extract place names using improved regex
            places = list(set(re.findall(r"-\s+([A-Za-z\s]+)(?::|$)", response_text)))
            vr_links = self.get_vr_links(places)
            
            # Add VR links directly to itinerary items
            formatted_response = []
            for line in response_text.split("\n"):
                clean_line = re.sub(r"\[.*?\]", "", line)  # Remove any existing markdown links
                place_match = re.match(r"-\s+([A-Za-z\s]+)(?::|$)", clean_line)
                if place_match:
                    place = place_match.group(1).strip()
                    if place in vr_links:
                        clean_line += f" [YouTube]({vr_links[place]})"
                formatted_response.append(clean_line)
            
            itinerary_items = []
            for line in formatted_response:
                item = {'text': line.strip()}
                place_match = re.match(r"-\s+([A-Za-z\s]+)(?::|$)", line)
                if place_match:
                    place = place_match.group(1).strip()
                    if place in vr_links and vr_links[place] != "No link found":
                        item['youtube_url'] = vr_links[place]
                itinerary_items.append(item)
            return itinerary_items
        except Exception as e:
            print(f"Error generating itinerary: {str(e)}")
            return [{"text": "Failed to generate itinerary. Please try again."}]

class UserInput(BaseModel):
    location: str
    hotel: str
    days: int
    budget: str
    trip_type: str

@app.post("/generate_itinerary")
async def generate_itinerary_endpoint(user_input: UserInput) -> Dict:
    planner = TripPlanner()
    try:
        itinerary = planner.generate_itinerary(user_input.dict())
        return {"itinerary": itinerary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
