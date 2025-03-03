from django import forms
from .models import ItemLost, ItemFound
from django.core.validators import FileExtensionValidator

class BaseItemForm(forms.ModelForm):
    """Base form for both lost and found items."""

    class Meta:
        model = ItemLost  # Default, but will be overridden in subclasses
        fields = ['title', 'description', 'image', 'status', 'latitude', 'longitude', 'address']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter item title'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Provide a description'}),
            'status': forms.Select(attrs={'class': 'form-control'}),
            'latitude': forms.TextInput(attrs={'class': 'form-control', 'readonly': True}),
            'longitude': forms.TextInput(attrs={'class': 'form-control', 'readonly': True}),
            'address': forms.TextInput(attrs={'class': 'form-control', 'readonly': True, 'placeholder': 'Auto-filled address'}),
        }

    image = forms.ImageField(
        required=True,
        validators=[FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png'])],
        widget=forms.ClearableFileInput(attrs={'class': 'form-control-file'})
    )

    def clean_title(self):
        """Ensure the title is meaningful and not too short."""
        title = self.cleaned_data.get('title')
        if len(title) < 3:
            raise forms.ValidationError("Title must be at least 3 characters long.")
        return title

    def clean_description(self):
        """Ensure a description is provided."""
        description = self.cleaned_data.get('description')
        if not description:
            raise forms.ValidationError("Please provide a description.")
        return description

# Subclass for Lost Items
class ItemLostForm(BaseItemForm):
    image = forms.ImageField(  
        required=False,  # ✅ Image is required for lost items  
        validators=[FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png'])],
        widget=forms.ClearableFileInput(attrs={'class': 'form-control-file'})
    )
    class Meta(BaseItemForm.Meta):
        model = ItemLost

# Subclass for Found Items
class ItemFoundForm(BaseItemForm):
    class Meta(BaseItemForm.Meta):
        model = ItemFound


from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import CustomUser

class SignupForm(UserCreationForm):
    """Form for user registration."""
    email = forms.EmailField(required=True)
    profile_picture = forms.ImageField(required=False)
    contact_number = forms.CharField(max_length=15, required=False)

    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2', 'profile_picture', 'contact_number']


class LoginForm(forms.Form):
    """Form for user login."""
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput)
