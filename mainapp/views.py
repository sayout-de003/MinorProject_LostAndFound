from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth import login as auth_login, logout as auth_logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Q
from django.db import models
from .models import ItemLost, ItemFound, CustomUser,ChatMessage
from .forms import ItemLostForm, ItemFoundForm, SignupForm, LoginForm, ProfileUpdateForm
from .utils import (
    store_image_features, search_similar_images, compute_text_similarity,
    calculate_distance, get_user_location, generate_image_caption, extract_text_from_image,
    get_matching_weights, calculate_time_decay
)

# Homepage
def home(request):
    return render(request, 'mainapp/home.html')

# Profile management
@login_required(login_url='login')
def profile(request):
    user = request.user
    lost_items = ItemLost.objects.filter(user=user).order_by('-created_at')
    found_items = ItemFound.objects.filter(user=user).order_by('-created_at')
    
    if request.method == "POST":
        form = ProfileUpdateForm(request.POST, request.FILES, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile updated successfully!")
            return redirect('profile')
        else:
            messages.error(request, "Error updating profile.")
    else:
        form = ProfileUpdateForm(instance=user)
    
    context = {'form': form, 'lost_items': lost_items, 'found_items': found_items}
    return render(request, 'mainapp/profile.html', context)

# Update item status
@login_required(login_url='login')
def update_item_status(request, item_id, item_type):
    if item_type == 'lost':
        item = get_object_or_404(ItemLost, id=item_id, user=request.user)
    else:
        item = get_object_or_404(ItemFound, id=item_id, user=request.user)
    
    if request.method == "POST":
        item.resolve_item()
        messages.success(request, f"{item.title} marked as resolved.")
        return redirect('profile')
    
    return render(request, 'mainapp/confirm_resolve.html', {'item': item, 'item_type': item_type})

# Authentication views
def signup(request):
    if request.method == "POST":
        form = SignupForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            messages.success(request, "Signup successful!")
            return redirect('home')
        else:
            messages.error(request, "Error in signup.")
    else:
        form = SignupForm()
    return render(request, 'mainapp/signup.html', {'form': form})

def login(request):
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
                messages.error(request, "Invalid credentials.")
    else:
        form = LoginForm()
    return render(request, 'mainapp/login.html', {'form': form})

@login_required
def logout(request):
    auth_logout(request)
    messages.success(request, "You have been logged out.")
    return redirect('home')

# Item upload views
@login_required(login_url='login')
def upload_item_lost(request):
    if request.method == "POST":
        form = ItemLostForm(request.POST, request.FILES)
        if form.is_valid():
            item = form.save(commit=False)
            item.user = request.user
            item.save()
            
            if not item.description and item.image:
                item.description = generate_ai_description(item.image.path)
                item.save()
            
            if item.image:
                store_image_features(item.image.path)
            
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
    if request.method == "POST":
        form = ItemFoundForm(request.POST, request.FILES)
        if form.is_valid():
            item = form.save(commit=False)
            item.user = request.user
            item.save()
            
            if not item.description and item.image:
                item.description = generate_ai_description(item.image.path)
                item.save()
            
            if item.image:
                store_image_features(item.image.path)
            
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

# Matching items
@login_required(login_url='login')
def match_items(request, item_id, item_type):
    if item_type == 'lost':
        item = get_object_or_404(ItemLost, id=item_id)
        opposite_items = ItemFound.objects.filter(status="Open")
    else:
        item = get_object_or_404(ItemFound, id=item_id)
        opposite_items = ItemLost.objects.filter(status="Open")
    
    weights = get_matching_weights(request.user)
    matches = []
    
    for match in opposite_items:
        img_score = 0
        if item.image and match.image:
            indices, distances = search_similar_images(item.image.path, k=1)
            if distances.size > 0:
                img_score = max(0, 1 - distances[0] / 1000)  # Normalize based on typical distance
        
        text_score = compute_text_similarity(item.description or "", match.description or "")
        
        location_score = 0
        if item.latitude and match.latitude:
            distance = calculate_distance(item.latitude, item.longitude, match.latitude, match.longitude)
            location_score = max(0, 1 / (1 + distance * 0.1))
        
        category_score = 1.0 if item.category == match.category else 0.0
        time_factor = calculate_time_decay(match, weights.time_decay_factor)
        
        overall_score = (
            weights.image_weight * img_score +
            weights.text_weight * text_score +
            weights.location_weight * location_score +
            weights.category_weight * category_score
        ) * time_factor
        
        matches.append({"match": match, "score": overall_score})
    
    matches = sorted(matches, key=lambda x: x["score"], reverse=True)[:10]
    
    if not matches or matches[0]["score"] < 0.3:
        return render(request, "mainapp/no_match.html")
    return render(request, 'mainapp/matches.html', {'item': item, 'matches': matches})

# Resolve item
@login_required(login_url='login')
def resolve_item(request, item_id, item_type):
    if item_type == 'lost':
        item = get_object_or_404(ItemLost, id=item_id)
    else:
        item = get_object_or_404(ItemFound, id=item_id)
    
    item.status = "Resolved"
    item.save()
    item.delete()
    return redirect('home')

# Location and description utilities
@login_required(login_url='login')
def get_location(request):
    lat = request.GET.get("lat")
    lon = request.GET.get("lon")
    if lat and lon:
        address = get_user_location(float(lat), float(lon))[2]
        return JsonResponse({"address": address})
    return JsonResponse({"error": "Invalid coordinates"}, status=400)

@csrf_exempt
def generate_description(request):
    if request.method == "POST" and request.FILES.get("image"):
        img = request.FILES["image"]
        description = generate_ai_description(img)
        return JsonResponse({"description": description})
    return JsonResponse({"error": "Invalid request"}, status=400)

def generate_ai_description(image_path):
    caption = generate_image_caption(image_path)
    text = extract_text_from_image(image_path)
    return f"Image Caption: {caption}. Detected Text: {text}" if text != "No text detected" else f"Image Caption: {caption}."

# Item detail
def item_detail(request, item_id):
    item = get_object_or_404(ItemLost, id=item_id)  # Adjust for ItemFound if needed
    return render(request, "mainapp/item_detail.html", {"item": item})

# Search items
@login_required(login_url='login')
def search_items(request):
    query = request.GET.get("query", "")
    category = request.GET.get("category", "")
    location = request.GET.get("location", "")
    sort_by = request.GET.get("sort_by", "date")
    
    lost_items = ItemLost.objects.filter(status="Open")
    found_items = ItemFound.objects.filter(status="Open")
    
    if query:
        lost_items = lost_items.filter(Q(title__icontains=query) | Q(description__icontains=query))
        found_items = found_items.filter(Q(title__icontains=query) | Q(description__icontains=query))
    if category:
        lost_items = lost_items.filter(category=category)
        found_items = found_items.filter(category=category)
    if location:
        lost_items = lost_items.filter(address__icontains=location)
        found_items = found_items.filter(address__icontains=location)
    
    items = list(lost_items) + list(found_items)
    
    if sort_by == "date":
        items.sort(key=lambda x: x.created_at, reverse=True)
    elif sort_by == "location" and request.user.latitude and request.user.longitude:
        items.sort(key=lambda x: calculate_distance(
            x.latitude or 0, x.longitude or 0, request.user.latitude, request.user.longitude
        ))
    
    return render(request, "mainapp/search_results.html", {"items": items, "query": query})

# Update matching weights
@login_required(login_url='login')
def update_weights(request):
    weights = get_matching_weights(request.user)
    if request.method == "POST":
        weights.image_weight = float(request.POST.get("image_weight", weights.image_weight))
        weights.text_weight = float(request.POST.get("text_weight", weights.text_weight))
        weights.location_weight = float(request.POST.get("location_weight", weights.location_weight))
        weights.category_weight = float(request.POST.get("category_weight", weights.category_weight))
        weights.time_decay_factor = float(request.POST.get("time_decay_factor", weights.time_decay_factor))
        weights.save()
        messages.success(request, "Matching weights updated successfully!")
        return redirect('profile')
    return render(request, 'mainapp/update_weights.html', {'weights': weights})

# Autocomplete search
@login_required(login_url='login')
def autocomplete_search(request):
    query = request.GET.get("query", "")
    if query:
        suggestions = ItemLost.objects.filter(
            Q(title__icontains=query) | Q(description__icontains=query)
        ).union(ItemFound.objects.filter(
            Q(title__icontains=query) | Q(description__icontains=query)
        )).values_list('title', flat=True)[:5]
        return JsonResponse({"suggestions": list(suggestions)})
    return JsonResponse({"suggestions": []})

def get_chat_room_id(user1, user2, item_lost_id, item_found_id):
    """Generate consistent room ID regardless of sender/receiver order."""
    ids = sorted([user1.id, user2.id])
    return f"{ids[0]}_{ids[1]}_{item_lost_id}_{item_found_id}"


from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User

@login_required
def chat_view(request, room_name, receiver_id):
    receiver = get_object_or_404(User, id=receiver_id)
    
    context = {
        'room_name': room_name,
        'receiver': receiver,
        'current_user': request.user,
    }
    
    return render(request, 'mainapp/chat.html', context)