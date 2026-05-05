import joblib
import pandas as pd
from django.shortcuts import render


model = joblib.load("smart_irrigation/models/best_model.pkl")
scaler = joblib.load("smart_irrigation/models/scaler.pkl")
encoder = joblib.load("smart_irrigation/models/onehot_encoder.pkl")
label_encoder = joblib.load("smart_irrigation/models/label_encoder.pkl")


def predict_irrigation(request):
    prediction_result = None

    if request.method == "POST":
        data = {
            "Soil_Type": request.POST.get("Soil_Type"),
            "Crop_Type": request.POST.get("Crop_Type"),
            "Crop_Growth_Stage": request.POST.get("Crop_Growth_Stage"),
            "Season": request.POST.get("Season"),
            "Irrigation_Type": request.POST.get("Irrigation_Type"),
            "Water_Source": request.POST.get("Water_Source"),
            "Mulching_Used": request.POST.get("Mulching_Used"),
            "Region": request.POST.get("Region"),

            "Soil_pH": float(request.POST.get("Soil_pH")),
            "Soil_Moisture": float(request.POST.get("Soil_Moisture")),
            "Organic_Carbon": float(request.POST.get("Organic_Carbon")),
            "Electrical_Conductivity": float(request.POST.get("Electrical_Conductivity")),
            "Temperature_C": float(request.POST.get("Temperature_C")),
            "Humidity": float(request.POST.get("Humidity")),
            "Rainfall_mm": float(request.POST.get("Rainfall_mm")),
            "Sunlight_Hours": float(request.POST.get("Sunlight_Hours")),
            "Wind_Speed_kmh": float(request.POST.get("Wind_Speed_kmh")),
            "Field_Area_hectare": float(request.POST.get("Field_Area_hectare")),
            "Previous_Irrigation_mm": float(request.POST.get("Previous_Irrigation_mm")),
        }

        df = pd.DataFrame([data])

        categorical_cols = [
            "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
            "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"
        ]

        numerical_cols = [
            "Soil_pH", "Soil_Moisture", "Organic_Carbon",
            "Electrical_Conductivity", "Temperature_C", "Humidity",
            "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh",
            "Field_Area_hectare", "Previous_Irrigation_mm"
        ]

        df_num = df[numerical_cols]
        df_cat = df[categorical_cols]

        df_num_scaled = pd.DataFrame(
            scaler.transform(df_num),
            columns=numerical_cols
        )

        df_cat_encoded = pd.DataFrame(
            encoder.transform(df_cat),
            columns=encoder.get_feature_names_out(categorical_cols)
        )

        final_input = pd.concat([df_num_scaled, df_cat_encoded], axis=1)

        prediction = model.predict(final_input)
        prediction_result = label_encoder.inverse_transform(prediction)[0]

    return render(request, "index.html", {"prediction": prediction_result})