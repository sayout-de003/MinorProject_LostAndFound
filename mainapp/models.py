from django.db import models
from django.contrib.auth.models import AbstractUser, Group, Permission
from django.utils.timezone import now

# Status choices for items
STATUS_CHOICES = [
    ('Open', 'Open'),
    ('Resolved', 'Resolved')
]

class CustomUser(AbstractUser):
    """Extended user model with additional fields."""
    profile_picture = models.ImageField(upload_to='profiles/', blank=True, null=True)
    contact_number = models.CharField(max_length=15, blank=True)

    # Fixing reverse accessor clashes
    groups = models.ManyToManyField(
        Group, related_name="customuser_groups", blank=True
    )
    user_permissions = models.ManyToManyField(
        Permission, related_name="customuser_permissions", blank=True
    )

    def __str__(self):
        return self.username


class BaseItem(models.Model):
    """Abstract base model for lost and found items."""
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)  # Use CustomUser
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to='items/')
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    address = models.CharField(max_length=255, blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Open')
    created_at = models.DateTimeField(default=now)

    class Meta:
        abstract = True  # This makes it an abstract base class

    def resolve_item(self):
        """Marks the item as resolved and deletes it."""
        self.status = "Resolved"
        self.save()
        self.delete()  # Remove from DB after resolution

    def __str__(self):
        return f"{self.title} ({self.status})"


class ItemLost(BaseItem):
    """Model for lost items."""
    pass


class ItemFound(BaseItem):
    """Model for found items."""
    matched_item = models.ForeignKey(ItemLost, null=True, blank=True, on_delete=models.SET_NULL)
