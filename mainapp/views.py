from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.utils.timezone import now
from .models import ItemLost, ItemFound
from .forms import ItemLostForm, ItemFoundForm
from .utils import (
    store_image_features, search_similar_images, compute_text_similarity,
    calculate_distance, get_user_location, generate_image_caption, extract_text_from_image
)
from PIL import Image


def home(request):
    """Render the homepage."""
    return render(request, 'mainapp/home.html')


from django.shortcuts import render, redirect
from django.contrib.auth import login as auth_login, logout as auth_logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import CustomUser
from .forms import SignupForm, LoginForm

def signup(request):
    """Handles user registration."""
    if request.method == "POST":
        form = SignupForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            messages.success(request, "Signup successful!")
            return redirect('home')
        else:
            messages.error(request, "Error in signup. Please check the form.")
    else:
        form = SignupForm()
    
    return render(request, 'mainapp/signup.html', {'form': form})


def login(request):
    """Handles user login."""
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user:
                auth_login(request, user)
                messages.success(request, "Login successful!")
                return redirect('home')
            else:
                messages.error(request, "Invalid credentials. Please try again.")
    else:
        form = LoginForm()

    return render(request, 'mainapp/login.html', {'form': form})


@login_required
def logout(request):
    """Handles user logout."""
    auth_logout(request)
    messages.success(request, "You have been logged out.")
    return redirect('home')

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import ItemLost, CustomUser
from .forms import ItemLostForm
from .utils import  store_image_features, get_user_location

@login_required(login_url='login')
def upload_item_lost(request):
    """Handles uploading lost items."""
    if request.method == "POST":
        form = ItemLostForm(request.POST, request.FILES)
        if form.is_valid():
            item = form.save(commit=False)  # Prevent immediate saving
            item.user = get_object_or_404(CustomUser, id=request.user.id)  # Assign logged-in user

            item.save()

            # **Generate a description only if no description was provided**
            if not item.description and item.image:
                item.description = generate_ai_description(item.image.path)
                item.save()

            # **Store image features only if an image is uploaded**
            if item.image:
                store_image_features(item.image.path)

            # **Handle auto-location feature**
            if "auto_location" in request.POST:
                lat, lon, address = get_user_location()
                if lat and lon:
                    item.latitude = lat
                    item.longitude = lon
                    item.address = address
                    item.save()

            return redirect('match_items', item.id, 'lost')

    else:
        form = ItemLostForm()
    
    return render(request, 'mainapp/upload_lost.html', {'form': form})


@login_required(login_url='login')
def upload_item_found(request):
    """Handles uploading found items."""
    if request.method == "POST":
        form = ItemFoundForm(request.POST, request.FILES)
        if form.is_valid():
            item = form.save(commit=False)  # Prevent immediate saving
            item.user = get_object_or_404(CustomUser, id=request.user.id)
  # Assign the logged-in user
            print(type(request.user))  # Check if it's <class 'mainapp.models.CustomUser'> or <class 'django.contrib.auth.models.User'>

            item.save()

            # Auto-generate a description if none is provided
            if not item.description:
                item.description = generate_ai_description(item.image.path)
                item.save()

            # Store image features for search
            store_image_features(item.image.path)

            # If "Auto Location" was clicked
            if "auto_location" in request.POST:
                lat, lon, address = get_user_location()
                if lat and lon:
                    item.latitude = lat
                    item.longitude = lon
                    item.address = address
                    item.save()

            return redirect('match_items', item.id, 'found')

    else:
        form = ItemFoundForm()
    return render(request, 'mainapp/upload_found.html', {'form': form})

from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, render
from .models import ItemLost, ItemFound
from .utils import search_similar_images, compute_text_similarity, calculate_distance

from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required

@login_required(login_url='login')
def match_items(request, item_id, item_type):
    """Matches lost items with found items."""
    
    if item_type == 'lost':
        item = get_object_or_404(ItemLost, id=item_id)
        opposite_items = ItemFound.objects.filter(status="Open")
    else:
        item = get_object_or_404(ItemFound, id=item_id)
        opposite_items = ItemLost.objects.filter(status="Open")

    matches = []

    for match in opposite_items:
        img_score = 0  # Default image score

        if item.image and match.image:
            image_indices, img_distances = search_similar_images(item.image.path)
            if img_distances is not None and len(img_distances) > 0:
                img_score = (1 - img_distances[0])  # Normalize score

        # Text similarity score (0 to 1)
        text_score = compute_text_similarity(item.description, match.description)

        # Location similarity score (normalized)
        if item.latitude and item.longitude and match.latitude and match.longitude:
            distance = calculate_distance(item.latitude, item.longitude, match.latitude, match.longitude)
            location_score = 1 / (1 + distance)  # Normalize to 0-1 scale
        else:
            location_score = 0  # Default if no location data

        # Weighted scoring system (image: 50%, text: 30%, location: 20%)
        overall_score = img_score * 0.5 + text_score * 0.3 + location_score * 0.2

        matches.append({
            "match": match,
            "score": overall_score
        })

    # Sort matches by highest score
    matches = sorted(matches, key=lambda x: x["score"], reverse=True)

    # ✅ Fix: Ensure the function always returns a response
    if not matches:
        #return HttpResponse("No matching items found.", content_type="text/plain")
        return render(request, "mainapp/no_match.html")


    return render(request, 'mainapp/matches.html', {'item': item, 'matches': matches})


@login_required(login_url='login')
def resolve_item(request, item_id, item_type):
    """Marks an item as resolved and removes it from the database."""
    if item_type == 'lost':
        item = get_object_or_404(ItemLost, id=item_id)
    else:
        item = get_object_or_404(ItemFound, id=item_id)

    item.status = "Resolved"
    item.save()
    item.delete()

    return redirect('home')

@login_required(login_url='login')
def get_location(request):
    """Fetches an address based on latitude and longitude."""
    lat = request.GET.get("lat")
    lon = request.GET.get("lon")

    if lat and lon:
        address = get_user_location(float(lat), float(lon))
        return JsonResponse({"address": address})

    return JsonResponse({"error": "Invalid coordinates"}, status=400)


@csrf_exempt
def generate_description(request):
    """Generates AI-based descriptions for an image."""
    if request.method == "POST" and request.FILES.get("image"):
        img = request.FILES["image"]
        description = generate_ai_description(img)

        return JsonResponse({"description": description})

    return JsonResponse({"error": "Invalid request"}, status=400)


def generate_ai_description(image_path):
    """Combines AI-based captioning and OCR for detailed description."""
    image_caption = generate_image_caption(image_path)
    extracted_text = extract_text_from_image(image_path)

    if extracted_text != "No text detected":
        description = f"Image Caption: {image_caption}. Detected Text: {extracted_text}"
    else:
        description = f"Image Caption: {image_caption}."

    return description