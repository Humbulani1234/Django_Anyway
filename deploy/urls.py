from django.contrib import admin
from django.urls import path, include
from . import views


urlpatterns = [

    path('',views.inputs, name='deployment'),
    path('Home/',views.home, name='home')

]
