# Ce fichier contient la logique de traitement

from django.shortcuts import render
import joblib

# Charger le modèle ML
model = joblib.load("ml/model.pkl")

def predict(request):
    """
    Cette fonction gère :
    - l'affichage du formulaire
    - la prédiction après envoi
    """

    if request.method == "POST":
        # Récupération des données depuis le formulaire
        temperature = float(request.POST['temperature'])
        humidity = float(request.POST['humidity'])
        soil = float(request.POST['soil'])

        # Prédiction avec le modèle
        prediction = model.predict([[temperature, humidity, soil]])

        # Résultat lisible
        if prediction[0] == 1:
            result = "Irrigation nécessaire "
        else:
            result = "Pas besoin d'irrigation "

        return render(request, "result.html", {"result": result})

    # Afficher le formulaire au début
    return render(request, "form.html")