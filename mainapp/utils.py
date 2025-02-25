
import torch
import faiss
import numpy as np
import cv2
import requests
import googlemaps
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from PIL import Image
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer, util
import os
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration

# Constants
FAISS_INDEX_PATH = "image_features.index"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GMAPS_API_KEY =  "AIzaSyDV__fcHGKtsviQcNHHsrz3veK2KGg8P3s"#googlemaps.Client(key="AIzaSyDV__fcHGKtsviQcNHHsrz3veK2KGg8P3s")  # Replace with actual API key

# Load FAISS index
def initialize_faiss_index(dimension):
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        if index.d != dimension:
            print(f"Dimension mismatch! Reinitializing FAISS index with {dimension} dimensions.")
            index = faiss.IndexFlatL2(dimension)
    else:
        index = faiss.IndexFlatL2(dimension)
    return index

# Load ResNet Model for feature extraction
def load_model():
    model = models.resnet50(pretrained=True).to(DEVICE)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
    model.eval()
    return model

# Image Preprocessing
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# Extract Image Features
def extract_features(image_path, model):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        features = model(image_tensor).squeeze().cpu().numpy()
    return features.flatten()

# Store Image Features in FAISS
def store_image_features(image_path, model=None):
    if model is None:
        model = load_model()  # Ensure model is loaded

    features = extract_features(image_path, model)
    dimension = features.shape[0]
    index = initialize_faiss_index(dimension)
    index.add(np.array([features]))
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"Stored feature vector of shape {features.shape} in FAISS.")


# Search Similar Images
def search_similar_images(image_path, model, k=5):
    features = extract_features(image_path, model)
    index = initialize_faiss_index(features.shape[0])
    distances, indices = index.search(np.array([features]), k)
    return indices[0], distances[0]

# Text Similarity Model
text_model = SentenceTransformer('all-MiniLM-L6-v2')
def compute_text_similarity(text1, text2):
    embedding1 = text_model.encode(text1, convert_to_tensor=True)
    embedding2 = text_model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding1, embedding2).item()

# Google Maps API Client
gmaps = googlemaps.Client(key=GMAPS_API_KEY)

def get_location_details(lat, lon):
    result = gmaps.reverse_geocode((lat, lon))
    return result[0]["formatted_address"] if result else "Unknown location"

def get_coordinates(address):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(address)
    return (location.latitude, location.longitude) if location else (None, None)

# Distance Calculation
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# BLIP Model for Image Captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_image_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    caption_ids = caption_model.generate(**inputs)
    return processor.batch_decode(caption_ids, skip_special_tokens=True)[0]

# OCR for Text Extraction
def extract_text_from_image(image_path):
    image = Image.open(image_path).convert("RGB")
    text = pytesseract.image_to_string(image)
    return text.strip() if text else "No text detected"

# Get User Location
def get_user_location(lat,lan):
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        lat, lon = map(float, data["loc"].split(","))
        address = f"{data.get('city', 'Unknown')}, {data.get('region', 'Unknown')}"
        return lat, lon, address
    except Exception as e:
        print("Location fetch error:", e)
        return None, None, "Unknown"





# import torch
# import faiss
# import numpy as np
# import cv2
# import requests
# import googlemaps
# from geopy.geocoders import Nominatim
# from geopy.distance import geodesic
# from PIL import Image
# from torchvision import models, transforms
# from sentence_transformers import SentenceTransformer, util
# import os

# # Load FAISS Index
# faiss_index =   "image_features.index" #faiss.IndexFlatL2(512)

# # Load Siamese Model for image similarity
# # model = model"image_features.index"s.resnet18(pretrained=True)
# # model.fc = torch.nn.Identity()
# # model.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet18(pretrained=True).to(device)
# model.eval()


# # Load Sentence Transformer for text similarity
# text_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Google Maps API Client
# gmaps = googlemaps.Client(key="AIzaSyDV__fcHGKtsviQcNHHsrz3veK2KGg8P3s")

# # Geolocation Converter
# geolocator = Nominatim(user_agent="geoapiExercises")

# def load_model():
#     model = models.resnet50(pretrained=True)  # You can replace with any model (EfficientNet, ViT, etc.)
#     model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
#     model.eval()
#     return model

# def preprocess_image(image_path):
#     image = Image.open(image_path).convert("RGB")  # Convert to RGB (handles grayscale & RGBA)
    
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),  # Resize to a standard size (larger than 224)
#         transforms.CenterCrop(224),  # Crop to match ResNet input
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     return image

# # Extract feature vector from an image
# def extract_features(image_path, model):
#     image = preprocess_image(image_path)
    
#     with torch.no_grad():
#         features = model(image)

#     features = features.view(-1).numpy()  # Flatten feature vector
#     return features
# # Initialize FAISS index (handles dynamic dimensions)
# def initialize_faiss_index(dimension):
#     if os.path.exists(FAISS_INDEX_PATH):
#         index = faiss.read_index(FAISS_INDEX_PATH)
#         if index.d != dimension:
#             print(f"Dimension mismatch! Reinitializing FAISS index with {dimension} dimensions.")
#             index = faiss.IndexFlatL2(dimension)  # Reset index
#     else:
#         index = faiss.IndexFlatL2(dimension)
    
#     return index
# # Store image features in FAISS
# def store_image_features(image_path):
#     model = load_model()
#     features = extract_features(image_path, model)
    
#     dimension = features.shape[0]
#     index = initialize_faiss_index(dimension)

#     index.add(np.array([features]))  # Add feature vector to FAISS
#     faiss.write_index(index, FAISS_INDEX_PATH)  # Save FAISS index
#     print(f"Stored feature vector of shape {features.shape} in FAISS.")

# def extract_image_features(image_path):
#     img = Image.open(image_path).convert('RGB')
#     transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
#     img_tensor = transform(img).unsqueeze(0)
#     with torch.no_grad():
#         features = model(img_tensor).squeeze().numpy()
#     return features

# def store_image_features(image_path):
#     features = extract_image_features(image_path)
#     faiss_index.add(np.array([features]))
#     return features

# def search_similar_images(image_path, k=5):
#     features = extract_image_features(image_path)
#     distances, indices = faiss_index.search(np.array([features]), k)
#     return indices[0], distances[0]

# def search_similar_images_batch(feature_array, k=5):
#     distances, indices = faiss_index.search(feature_array, k)
#     return indices, distances


# def compute_text_similarity(text1, text2):
#     embedding1 = text_model.encode(text1, convert_to_tensor=True)
#     embedding2 = text_model.encode(text2, convert_to_tensor=True)
#     return util.pytorch_cos_sim(embedding1, embedding2).item()

# def calculate_distance(lat1, lon1, lat2, lon2):
#     return geodesic((lat1, lon1), (lat2, lon2)).km

# def get_location_details(lat, lon):
#     result = gmaps.reverse_geocode((lat, lon))
#     return result[0]["formatted_address"] if result else "Unknown location"

# def get_coordinates(address):
#     location = geolocator.geocode(address)
#     if location:
#         return location.latitude, location.longitude
#     return None, None


# from transformers import BlipProcessor, BlipForConditionalGeneration

# # Load BLIP Model for Image Captioning
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# def generate_image_caption(image_path):
#     """Generates a caption for the given image."""
#     image = Image.open(image_path).convert('RGB')
#     inputs = processor(image, return_tensors="pt")
#     caption_ids = caption_model.generate(**inputs)
#     caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
#     return caption

# def get_user_location(a,b):
#     """Fetches user's geolocation."""
#     try:
#         response = requests.get("https://ipinfo.io/json")
#         data = response.json()
#         lat, lon = map(float, data["loc"].split(","))
#         address = data.get("city", "Unknown") + ", " + data.get("region", "Unknown")
#         return lat, lon, address
#     except Exception as e:
#         print("Location fetch error:", e)
#         return None, None, "Unknown"
# import pytesseract
# from PIL import Image

# def extract_text_from_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     text = pytesseract.image_to_string(image)
#     return text.strip() if text else "No text detected"



# #Start new 

# # import torch
# # import faiss
# # import numpy as np
# # from PIL import Image
# # from torchvision import models, transforms
# # from sentence_transformers import SentenceTransformer, util
# # from geopy.distance import geodesic
# # import os
# # import pickle
# # from transformers import BlipProcessor, BlipForConditionalGeneration
# # import requests
# # import pytesseract

# # # Initialize FAISS index and item IDs
# # faiss_index = faiss.IndexFlatL2(512)
# # item_ids = []

# # # Device configuration (compatible with Mac M1)
# # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# # # Load pre-trained models
# # model = models.resnet18(pretrained=True).to(device)
# # model.eval()
# # text_model = SentenceTransformer('all-MiniLM-L6-v2')
# # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# # caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# # def extract_image_features(image_path):
# #     """Extract features from an image using ResNet18."""
# #     img = Image.open(image_path).convert('RGB')
# #     transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
# #     img_tensor = transform(img).unsqueeze(0).to(device)
# #     with torch.no_grad():
# #         features = model(img_tensor).squeeze().cpu().numpy()
# #     return features

# # def store_image_features(image_path, item_id):
# #     """Store image features in FAISS and maintain item ID mapping."""
# #     features = extract_image_features(image_path)
# #     faiss_index.add(np.array([features]))
# #     item_ids.append(item_id)
# #     faiss.write_index(faiss_index, "faiss_index.bin")
# #     with open("item_ids.pkl", "wb") as f:
# #         pickle.dump(item_ids, f)

# # def get_top_similar_items(image_path, k=10):
# #     """Find top similar items based on image features."""
# #     features = extract_image_features(image_path)
# #     distances, indices = faiss_index.search(np.array([features]), k)
# #     similar_item_ids = [item_ids[i] for i in indices[0]]
# #     return similar_item_ids, distances[0]

# # def compute_text_similarity(text1, text2):
# #     """Compute text similarity using Sentence Transformers."""
# #     embedding1 = text_model.encode(text1, convert_to_tensor=True)
# #     embedding2 = text_model.encode(text2, convert_to_tensor=True)
# #     return util.pytorch_cos_sim(embedding1, embedding2).item()

# # def calculate_distance(lat1, lon1, lat2, lon2):
# #     """Calculate distance between two points in kilometers."""
# #     return geodesic((lat1, lon1), (lat2, lon2)).km

# # def generate_image_caption(image_path):
# #     """Generate a caption for the image using BLIP."""
# #     image = Image.open(image_path).convert('RGB')
# #     inputs = processor(image, return_tensors="pt").to(device)
# #     caption_ids = caption_model.generate(**inputs)
# #     caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
# #     return caption

# # def get_user_location():
# #     """Fetch user's geolocation using IP address."""
# #     try:
# #         response = requests.get("https://ipinfo.io/json")
# #         data = response.json()
# #         lat, lon = map(float, data["loc"].split(","))
# #         address = f"{data.get('city', 'Unknown')}, {data.get('region', 'Unknown')}"
# #         return lat, lon, address
# #     except Exception:
# #         return None, None, "Unknown"

# # def get_location_details(lat, lon):
# #     """Reverse geocode coordinates to get address (using Nominatim for free usage)."""
# #     from geopy.geocoders import Nominatim
# #     geolocator = Nominatim(user_agent="lost_found_app")
# #     location = geolocator.reverse((lat, lon))
# #     return location.address if location else "Unknown location"

# # def initialize_faiss():
# #     """Initialize FAISS index from existing items or saved files."""
# #     global faiss_index, item_ids
# #     if os.path.exists("faiss_index.bin") and os.path.exists("item_ids.pkl"):
# #         faiss_index = faiss.read_index("faiss_index.bin")
# #         with open("item_ids.pkl", "rb") as f:
# #             item_ids = pickle.load(f)
# #     else:
# #         faiss_index = faiss.IndexFlatL2(512)
# #         item_ids = []
# #         from .models import Item
# #         for item in Item.objects.all():
# #             features = extract_image_features(item.image.path)
# #             faiss_index.add(np.array([features]))
# #             item_ids.append(item.id)
# #         faiss.write_index(faiss_index, "faiss_index.bin")
# #         with open("item_ids.pkl", "wb") as f:
# #             pickle.dump(item_ids, f)





# # def extract_text_from_image(image_path):
# #     image = Image.open(image_path).convert("RGB")
# #     text = pytesseract.image_to_string(image)
# #     return text.strip() if text else "No text detected"



# # import torch
# # import clip
# # from PIL import Image


# # model, preprocess = clip.load("ViT-B/32", device=device)

# # # Define Categories
# # ITEM_CATEGORIES = [
# #     "ID Card", "Wallet", "Phone", "Keys", "Passport", "Book", "Credit Card", "Laptop", "Headphones"
# # ]

# # def detect_item_type(image_path):
# #     """
# #     Predicts the type of item in the image using CLIP.

# #     Args:
# #         image_path (str): Path to the uploaded image.

# #     Returns:
# #         str: Predicted item type (e.g., "Wallet", "Phone").
# #     """
# #     try:
# #         # Preprocess Image
# #         image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# #         # Encode Text & Image
# #         text_inputs = clip.tokenize(ITEM_CATEGORIES).to(device)
# #         with torch.no_grad():
# #             image_features = model.encode_image(image)
# #             text_features = model.encode_text(text_inputs)

# #         # Compute Similarity
# #         similarity = (image_features @ text_features.T).softmax(dim=-1)
# #         predicted_index = similarity.argmax().item()
# #         predicted_label = ITEM_CATEGORIES[predicted_index]

# #         return predicted_label

# #     except Exception as e:
# #         print(f"Error in detect_item_type: {e}")
# #         return "Unknown Item"


