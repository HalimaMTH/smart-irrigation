# Ce fichier gère les routes spécifiques à l'application

from django.urls import path
from . import views   # importer depuis le même dossier (correct)

urlpatterns = [
    path('', views.predict, name='predict'),
]