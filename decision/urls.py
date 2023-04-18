from django.contrib import admin
from django.urls import path, include
from . import views


urlpatterns = [

    path('',views.tree, name='decision'),
    # path('answer/',views.result, name='results')

]
