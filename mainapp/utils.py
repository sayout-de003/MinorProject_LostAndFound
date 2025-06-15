import torch
import faiss
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer, util
from geopy.distance import geodesic
from django.utils.timezone import now
from .models import MatchingWeights
import googlemaps
from geopy.geocoders import Nominatim
import requests
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
# from django.contrib.gis.geos import Point

FAISS_INDEX_PATH = "image_features.index"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GMAPS_API_KEY = "AIzaSyDV__fcHGKtsviQcNHHsrz3veK2KGg8P3s"
DIMENSION = 1280

def load_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).to(DEVICE)
    model.classifier = torch.nn.Identity()
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def extract_features(image_path, model):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        features = model(image_tensor).squeeze().cpu().numpy()
    return features.flatten()

def initialize_faiss_index(dimension=DIMENSION):
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        if index.d != dimension:
            index = faiss.IndexFlatL2(dimension)
    else:
        index = faiss.IndexFlatL2(dimension)
    return index

def store_image_features(image_path, model=None):
    if model is None:
        model = load_model()
    features = extract_features(image_path, model)
    if features.shape[0] != DIMENSION:
        raise ValueError(f"Feature dimension {features.shape[0]} does not match expected {DIMENSION}")
    index = initialize_faiss_index()
    index.add(np.array([features]))
    faiss.write_index(index, FAISS_INDEX_PATH)

def search_similar_images(image_path, model=None, k=5):
    if model is None:
        model = load_model()
    features = extract_features(image_path, model)
    index = initialize_faiss_index()
    if index.ntotal == 0:
        return [], []
    distances, indices = index.search(np.array([features]), k)
    return indices[0], distances[0]

TEXT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
def compute_text_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    embedding1 = TEXT_MODEL.encode(text1, convert_to_tensor=True)
    embedding2 = TEXT_MODEL.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    return similarity if similarity > 0.5 else 0.0

# gmaps = googlemaps.Client(key=GMAPS_API_KEY)

# def get_location_details(lat, lon):
#     result = gmaps.reverse_geocode((lat, lon))
#     return result[0]["formatted_address"] if result else "Unknown location"

import time
from geopy.geocoders import Nominatim

def get_location_details(lat, lon):
    geolocator = Nominatim(user_agent="lostfoundapp")
    try:
        time.sleep(1)  # To respect Nominatim's usage policy
        print(f"Attempting reverse geocode: lat={lat}, lon={lon}")  # Debug
        location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
        print(f"Resolved address: {location.address if location else 'None'}")  # Debug
        return location.address if location else "Unknown location"
    except Exception as e:
        print(f"Geocoding error: {e}")
        return "Unknown location"


def get_coordinates(address):
    geolocator = Nominatim(user_agent="lostfoundapp")
    location = geolocator.geocode(address)
    return (location.latitude, location.longitude) if location else (None, None)

def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

def generate_image_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    caption_ids = caption_model.generate(**inputs)
    return processor.batch_decode(caption_ids, skip_special_tokens=True)[0]

def extract_text_from_image(image_path):
    image = Image.open(image_path).convert("RGB")
    text = pytesseract.image_to_string(image)
    return text.strip() if text else "No text detected"

def get_user_location(lat=None, lon=None):
    if lat and lon:
        address = get_location_details(lat, lon)
        return lat, lon, address
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        lat, lon = map(float, data["loc"].split(","))
        address = f"{data.get('city', 'Unknown')}, {data.get('region', 'Unknown')}"
        return lat, lon, address
    except Exception:
        return None, None, "Unknown"

def get_matching_weights(user=None):
    if user:
        weights, created = MatchingWeights.objects.get_or_create(user=user)
    else:
        weights, created = MatchingWeights.objects.get_or_create(user__isnull=True)
    return weights

def calculate_time_decay(item, decay_factor):
    days_old = (now() - item.created_at).days
    return max(0.1, 1 - days_old * decay_factor)