from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, ItemLost, ItemFound

# Extend the UserAdmin to include CustomUser fields
class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = ('username', 'email', 'contact_number', 'is_staff', 'is_active')
    fieldsets = (
        ("Personal Info", {"fields": ('username', 'email', 'first_name', 'last_name', 'contact_number', 'profile_picture')}),
        ("Permissions", {"fields": ('is_staff', 'is_active', 'groups', 'user_permissions')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'password1', 'password2', 'contact_number', 'profile_picture', 'is_staff', 'is_active')
        }),
    )
    search_fields = ('username', 'email', 'contact_number')
    ordering = ('username',)

# Registering BaseItem-derived models with extra functionalities
class BaseItemAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'status', 'created_at', 'latitude', 'longitude')
    list_filter = ('status', 'created_at')
    search_fields = ('title', 'description', 'user__username', 'address')
    readonly_fields = ('created_at',)

class ItemLostAdmin(BaseItemAdmin):
    pass

class ItemFoundAdmin(BaseItemAdmin):
    list_display = ('title', 'user', 'status', 'created_at', 'matched_item')
    autocomplete_fields = ('matched_item',)

# Register the models to the admin site
admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(ItemLost, ItemLostAdmin)
admin.site.register(ItemFound, ItemFoundAdmin)
