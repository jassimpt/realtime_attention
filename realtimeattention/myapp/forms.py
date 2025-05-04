from django import forms
from .models import student_table

class StudentRegisterForm(forms.ModelForm):
    confirm_password = forms.CharField(widget=forms.PasswordInput(), label="Confirm Password")

    class Meta:
        model = student_table
        fields = ['name', 'email', 'phone_number', 'password']
        widgets = {
            'password': forms.PasswordInput(),
        }

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')

        if password and confirm_password and password != confirm_password:
            self.add_error('confirm_password', "Passwords do not match.")
        
        if student_table.objects.filter(email=cleaned_data.get('email')).exists():
            self.add_error('email', "Email is already registered.")

        if student_table.objects.filter(phone_number=cleaned_data.get('phone_number')).exists():
            self.add_error('phone_number', "Phone number is already registered.")