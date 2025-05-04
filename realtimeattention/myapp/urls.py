"""
URL configuration for realtimeattention project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from myapp import views

urlpatterns = [

    path('admin/',views.admin_login,name="admin_login"),
    path('admin/home/',views.admin_home,name="admin_home"),
    path('admin_logout/', views.admin_logout, name='admin_logout'),
    path('admin/student/list/',views.list_student,name="list_student"),
    path('admin/attendence/list/',views.list_attendence,name="list_attendence"),

    path('',views.student_login,name="student_login"),
    path('register/',views.student_register,name="student_register"),
    path('student_home/',views.student_home,name="student_home"),
    path('logout/', views.student_logout, name='student_logout'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('stop_video_feed/', views.stop_video_feed, name='stop_video_feed'),
    

]