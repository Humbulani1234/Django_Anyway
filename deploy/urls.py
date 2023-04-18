from django.contrib import admin
from django.urls import path, include
from . import views


urlpatterns = [

    path('',views.inputs, name='deployment'),
    # path('answer/',views.result, name='results')

]
