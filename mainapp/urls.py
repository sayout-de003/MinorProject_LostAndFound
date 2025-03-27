from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('signup/', views.signup, name='signup'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('upload/lost/', views.upload_item_lost, name='upload_item_lost'),
    path('upload/found/', views.upload_item_found, name='upload_item_found'),
    path('match/<int:item_id>/<str:item_type>/', views.match_items, name='match_items'),
    path('resolve/<int:item_id>/<str:item_type>/', views.resolve_item, name='resolve_item'),
    path('get-location/', views.get_location, name='get_location'),
    path('generate-description/', views.generate_description, name='generate_description'),
    path('search/', views.search_items, name='search_items'),
    path('item/<int:item_id>/', views.item_detail, name='item_detail'),
    path('profile/', views.profile, name='profile'),
    path('item/<int:item_id>/<str:item_type>/resolve/', views.update_item_status, name='update_item_status'),
]
   

