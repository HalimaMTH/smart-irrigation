from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_irrigation, name='predict_irrigation'),
]