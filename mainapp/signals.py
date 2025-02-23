from django.db.models.signals import post_save
from django.dispatch import receiver
from django.core.mail import send_mail
from .models import Item
from .utils import get_top_similar_items, compute_text_similarity, calculate_distance

@receiver(post_save, sender=Item)
def notify_matches(sender, instance, created, **kwargs):
    """Send notifications to owners of matching items."""
    if created:
        similar_item_ids, img_distances = get_top_similar_items(instance.image.path, k=10)
        id_to_distance = dict(zip(similar_item_ids, img_distances))
        opposite_status = 'Found' if instance.status == 'Lost' else 'Lost'
        potential_matches = Item.objects.filter(id__in=similar_item_ids, status=opposite_status)
        
        for match in potential_matches:
            distance = id_to_distance.get(match.id, float('inf'))
            image_similarity = 1 / (1 + distance)
            text_score = compute_text_similarity(instance.description, match.description)
            location_score = 1 / (1 + calculate_distance(
                instance.latitude or 0, instance.longitude or 0,
                match.latitude or 0, match.longitude or 0
            ))
            overall_score = 0.5 * image_similarity + 0.3 * text_score + 0.2 * location_score
            
            if overall_score > 0.7:
                send_mail(
                    subject='Potential Match Found',
                    message=f'A potential match for your {match.status.lower()} item has been reported: {instance.title}',
                    from_email='system@example.com',
                    recipient_list=[match.user.email],
                    fail_silently=False,
                )