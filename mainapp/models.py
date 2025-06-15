from django.db import models
from django.contrib.auth.models import AbstractUser, Group, Permission
from django.utils.timezone import now
from django.conf import settings
import json

# Status choices for items
STATUS_CHOICES = [
    ('Open', 'Open'),
    ('Resolved', 'Resolved')
]

CONDITION_CHOICES = [
    ('New', 'New'),
    ('Used', 'Used'),
    ('Damaged', 'Damaged'),
]

class CustomUser(AbstractUser):
    profile_picture = models.ImageField(upload_to='profiles/', blank=True, null=True)
    contact_number = models.CharField(max_length=15, blank=True)

    groups = models.ManyToManyField(Group, related_name="customuser_groups", blank=True)
    user_permissions = models.ManyToManyField(Permission, related_name="customuser_permissions", blank=True)

    def __str__(self):
        return self.username

class MatchingWeights(models.Model):
    """Model to store dynamic matching weights per user."""
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, null=True, blank=True)
    image_weight = models.FloatField(default=0.5)  # Default 50%
    text_weight = models.FloatField(default=0.3)   # Default 30%
    location_weight = models.FloatField(default=0.2)  # Default 20%
    category_weight = models.FloatField(default=0.1)  # New: 10% for category match
    time_decay_factor = models.FloatField(default=0.01)  # Rate of score decay per day

    class Meta:
        unique_together = ('user',)  # One set of weights per user

    def __str__(self):
        return f"Weights for {self.user or 'Default'}"

class BaseItem(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to='items/', blank=True, null=True)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    address = models.CharField(max_length=255, blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Open')
    created_at = models.DateTimeField(default=now)
    category = models.CharField(max_length=50, blank=True)  # e.g., "electronics", "jewelry"
    condition = models.CharField(max_length=10, choices=CONDITION_CHOICES, default='Used')
    tags = models.TextField(blank=True)  # Store tags as JSON string

    class Meta:
        abstract = True

    def get_tags(self):
        if not self.tags:
            return []
        return json.loads(self.tags)

    def set_tags(self, tags_list):
        self.tags = json.dumps(tags_list)
        self.save()

    def resolve_item(self):
        self.status = "Resolved"
        self.save()
        self.delete()

    def __str__(self):
        return f"{self.title} ({self.status})"

class ItemLost(BaseItem):
    pass

class ItemFound(BaseItem):
    matched_item = models.ForeignKey(ItemLost, null=True, blank=True, on_delete=models.SET_NULL)



 
    

from django.db import models
from django.contrib.auth.models import User
from .models import ItemLost, ItemFound

def user_directory_path(instance, filename):
    return 'user_{0}/chat/{1}'.format(instance.sender.id, filename)


class ChatMessage(models.Model):
    sender = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='sent_messages')
    receiver = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='received_messages')
    message = models.TextField(blank=True) #message can be optional since image/video can be sent
    image = models.ImageField(upload_to=user_directory_path, blank=True, null=True)
    video = models.FileField(upload_to=user_directory_path, blank=True, null=True)    
    timestamp = models.DateTimeField(auto_now_add=True)
    item_lost = models.ForeignKey(ItemLost, null=True, blank=True, on_delete=models.SET_NULL)
    item_found = models.ForeignKey(ItemFound, null=True, blank=True, on_delete=models.SET_NULL)
    is_read = models.BooleanField(default=False)
    

    def __str__(self):
        return f"{self.sender} to {self.receiver}: {self.message[:20]}"    