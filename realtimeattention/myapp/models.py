from django.db import models

# Create your models here.


class student_table(models.Model):
    name=models.CharField(max_length=100)
    email=models.CharField(max_length=100)
    phone_number=models.CharField(max_length=10)
    password = models.CharField(max_length=100,null=True,blank=True)


class emotion_table(models.Model):
    student_id = models.ForeignKey(student_table,on_delete=models.CASCADE,blank=True,null=True)
    Date = models.DateField(blank=True, null=True)
    emotion=models.CharField(max_length=50)
    emotion_score = models.CharField(max_length=100)
    movementup = models.CharField(max_length=100)
    movementdown = models.CharField(max_length=100)
    movementleft = models.CharField(max_length=100)
    movementright = models.CharField(max_length=100)
    attention_status = models.BooleanField(default=False)
    image = models.ImageField(upload_to='emotion_images/', blank=True, null=True) 
    

class Attendence(models.Model):
    student = models.ForeignKey(student_table, on_delete=models.CASCADE)
    is_attentive = models.BooleanField(default=True) 
    image = models.ImageField(upload_to='attendance_images/', null=True, blank=True) 
    timestamp = models.DateTimeField(auto_now_add=True) 

    def __str__(self):
        return self.student.name
    