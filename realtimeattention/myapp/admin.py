from django.contrib import admin
from myapp.models import student_table,emotion_table,Attendence

# Register your models here.

admin.site.register(student_table)
admin.site.register(emotion_table)
admin.site.register(Attendence)
