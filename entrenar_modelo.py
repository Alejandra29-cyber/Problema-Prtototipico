import pandas as pd
import json
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

print("🤖 Iniciando el entrenamiento de la RED NEURONAL...")

def limpiar_experiencia(texto):
    if isinstance(texto, str):
        if 'sin exper' in texto.lower(): return 'Sin Experiencia'
        if 'basic' in texto.lower(): return 'Basica'
        if 'intermedia' in texto.lower(): return 'Intermedia'
        if 'experta' in texto.lower(): return 'Experta'
    return 'Ninguna'

carpeta_datos = 'Data'
ruta_empleados_json = os.path.join(carpeta_datos, 'empleados.json')
ruta_modelo_nn_pkl = os.path.join(carpeta_datos, 'modelo_red_neuronal.pkl')
ruta_scaler_pkl = os.path.join(carpeta_datos, 'scaler.pkl')
ruta_encoder_pkl = os.path.join(carpeta_datos, 'label_encoder.pkl')
ruta_columnas_pkl = os.path.join(carpeta_datos, 'columnas_modelo_nn.pkl')

if not os.path.exists(carpeta_datos):
    os.makedirs(carpeta_datos)

with open(ruta_empleados_json, 'r', encoding='utf-8') as f:
    datos = json.load(f)

df = pd.DataFrame(datos['empleados'])
df = df[df['calidad_candidato'].notna() & (df['calidad_candidato'] != '')]

df.fillna('Ninguna', inplace=True)
df['experiencia_limpia'] = df['experiencia'].apply(limpiar_experiencia)
X_dummies = pd.get_dummies(df[['experiencia_limpia', 'licencias']], drop_first=True)
joblib.dump(X_dummies.columns, ruta_columnas_pkl)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dummies)
joblib.dump(scaler, ruta_scaler_pkl)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(df['calidad_candidato'])
joblib.dump(encoder, ruta_encoder_pkl)
print(f"   Etiquetas codificadas: {list(encoder.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

model_nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, activation='relu', solver='adam', early_stopping=True)
print("Entrenando la red neuronal...")
model_nn.fit(X_train, y_train)

accuracy = model_nn.score(X_test, y_test)
print(f"Precisión de la Red Neuronal: {accuracy * 100:.2f}%")

joblib.dump(model_nn, ruta_modelo_nn_pkl)
print(f"🧠 ¡Entrenamiento completado! Modelo guardado en '{ruta_modelo_nn_pkl}'.")
