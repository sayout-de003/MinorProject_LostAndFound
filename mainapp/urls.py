from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/lost/', views.upload_item_lost, name='upload_item_lost'),
    path('upload/found/', views.upload_item_found, name='upload_item_found'),
    path('match/<int:item_id>/<str:item_type>/', views.match_items, name='match_items'),
    path('resolve/<int:item_id>/<str:item_type>/', views.resolve_item, name='resolve_item'),
    path('get-location/', views.get_location, name='get_location'),
    path('generate-description/', views.generate_description, name='generate_description'),
]
