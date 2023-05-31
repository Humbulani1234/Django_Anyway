from django.contrib import admin
from django.urls import path, include
from . import views


urlpatterns = [

    path('',views.tree, name='decision'),
    path('Confusion/',views.confusion_decision, name='confusion'),
    path('Tree/',views.decision_tree, name='tree'),
    path('Cross/',views.cross_validate, name='cross_validate'),
    # path('answer/',views.result, name='results')

]
