# ==============================
# DATAOPS PIPELINE - SMART IRRIGATION
# ==============================

# 1. Importation des bibliothèques nécessaires
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE


# 2. Création des dossiers nécessaires
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)


# 3. Définition du chemin du dataset
dataset_path = "../data/raw/irrigation_prediction.csv"


# 4. Chargement du dataset
df = pd.read_csv(dataset_path)

print("Dataset chargé avec succès")
print(df.head())


# 5. Analyse générale du dataset
print("\nInformations du dataset :")
print(df.info())

print("\nStatistiques descriptives :")
print(df.describe())

print("\nValeurs manquantes :")
print(df.isnull().sum())


# 6. Suppression des observations doublantes
df = df.drop_duplicates()

print("\nAprès suppression des doublons :")
print(df.info())


# 7. Séparation des variables numériques et catégorielles
x_num = df.select_dtypes(include=[np.number])
x_str = df.select_dtypes(exclude=[np.number])

print("\nVariables numériques :")
print(x_num.info())

print("\nVariables catégorielles :")
print(x_str.info())


# 8. Imputation des variables numériques
imputer = IterativeImputer(random_state=43)

x_num_imputed = imputer.fit_transform(x_num)

imputed_x_num = pd.DataFrame(
    x_num_imputed,
    columns=x_num.columns
)

print("\nImputation numérique terminée")


# 9. Traitement des outliers avec IQR
def handle_outliers(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1

    Fb = Q1 - 1.5 * IQR
    Fh = Q3 + 1.5 * IQR

    return col.clip(lower=Fb, upper=Fh)


for col in imputed_x_num.columns:
    imputed_x_num[col] = handle_outliers(imputed_x_num[col])

print("\nTraitement des outliers terminé")


# 10. Imputation des variables catégorielles
for col in x_str.columns:
    x_str[col] = x_str[col].fillna(x_str[col].mode()[0])

print("\nImputation catégorielle terminée")
print(x_str.describe())


# 11. Séparation de la variable cible
y = x_str["Irrigation_Need"]
x_str = x_str.drop(columns="Irrigation_Need")


# 12. Concaténation des variables catégorielles et numériques
x = pd.concat([x_str, imputed_x_num], axis=1)

print("\nConcaténation terminée")
print(x.head())


# 13. Division des données en train et test
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=43
)

print("\nSplit train/test terminé")
print("x_train :", x_train.shape)
print("x_test :", x_test.shape)


# 14. Séparation num/cat pour train et test
x_num_train = x_train.select_dtypes(include=[np.number])
x_str_train = x_train.select_dtypes(exclude=[np.number])

x_num_test = x_test.select_dtypes(include=[np.number])
x_str_test = x_test.select_dtypes(exclude=[np.number])


# 15. Standardisation des variables numériques
scaler = StandardScaler()

x_train_s = scaler.fit_transform(x_num_train)
x_test_s = scaler.transform(x_num_test)

x_train_s = pd.DataFrame(
    x_train_s,
    columns=x_num_train.columns
)

x_test_s = pd.DataFrame(
    x_test_s,
    columns=x_num_test.columns
)

print("\nStandardisation terminée")


# 16. Encodage OneHot des variables catégorielles
encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

encoded_x_str_train = encoder.fit_transform(x_str_train)

encoded_x_str_df = pd.DataFrame(
    encoded_x_str_train,
    columns=encoder.get_feature_names_out(x_str_train.columns)
)

encoded_x_str_test = encoder.transform(x_str_test)

encoded_x_str_df_test = pd.DataFrame(
    encoded_x_str_test,
    columns=encoder.get_feature_names_out(x_str_test.columns)
)

print("\nEncodage OneHot terminé")


# 17. Concaténation finale train/test
x_train_final = pd.concat([x_train_s, encoded_x_str_df], axis=1)
x_test_final = pd.concat([x_test_s, encoded_x_str_df_test], axis=1)

print("\nDonnées finales prêtes")
print("x_train_final :", x_train_final.shape)
print("x_test_final :", x_test_final.shape)


# 18. Encodage de la variable cible y
label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

y_train_encoded = pd.DataFrame(
    y_train_encoded,
    columns=["Irrigation_Need"]
)

y_test_encoded = pd.DataFrame(
    y_test_encoded,
    columns=["Irrigation_Need"]
)

print("\nEncodage de y terminé")


# 19. Vérification du déséquilibre des classes
df_temp = pd.concat([x_train_final, y_train_encoded], axis=1)

print("\nDistribution avant SMOTE :")
print(df_temp["Irrigation_Need"].value_counts())


# 20. Application de SMOTE sur les données d'entraînement seulement
smote = SMOTE(random_state=43)

x_train_equ, y_train_equ = smote.fit_resample(
    x_train_final,
    y_train_encoded
)

print("\nDistribution après SMOTE :")
print(y_train_equ["Irrigation_Need"].value_counts())


# 21. Sauvegarde des données traitées
x_train_equ.to_csv("../data/processed/X_train.csv", index=False)
x_test_final.to_csv("../data/processed/X_test.csv", index=False)

y_train_equ.to_csv("../data/processed/y_train.csv", index=False)
y_test_encoded.to_csv("../data/processed/y_test.csv", index=False)


# 22. Sauvegarde des objets de preprocessing
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(encoder, "../models/onehot_encoder.pkl")
joblib.dump(label_encoder, "../models/label_encoder.pkl")
joblib.dump(imputer, "../models/imputer.pkl")


print("\nDataOps pipeline terminé avec succès")
print("Fichiers sauvegardés dans data/processed/")
print("Objets sauvegardés dans models/")