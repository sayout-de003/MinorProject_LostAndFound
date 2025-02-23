from django.db import models
from django.contrib.auth.models import User
from django.utils.timezone import now

STATUS_CHOICES = [
    ('Open', 'Open'),
    ('Resolved', 'Resolved')
]

class BaseItem(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
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
