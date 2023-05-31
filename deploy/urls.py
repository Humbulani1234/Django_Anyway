from django.contrib import admin
from django.urls import path, include
from . import views


urlpatterns = [

    path('',views.inputs, name='deployment'),
    path('Home/',views.home, name='home'),
    path('Roc/', views.roc, name='roc'),
    path('Confusion/', views.confusion_logistic, name='confusion'),

    path('Normal/', views.normal_plot, name='normal_plot'),
    path('Residuals/', views.residuals, name='residuals'),
    path('Partial/', views.partial, name='partial'),
    path('Student/', views.student, name='student'),
    path('Cooks/', views.cooks, name='cooks'),


]
