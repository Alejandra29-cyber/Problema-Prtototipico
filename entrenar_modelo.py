import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
import joblib

print("🤖 Iniciando el entrenamiento del modelo desde el archivo JSON...")

def limpiar_experiencia(texto):
    if isinstance(texto, str):
        if 'sin exper' in texto.lower():
            return 'Sin Experiencia'
        if 'basic' in texto.lower():
            return 'Basica'
        if 'intermedia' in texto.lower():
            return 'Intermedia'
        if 'experta' in texto.lower():
            return 'Experta'
    return 'Ninguna'

try:
    with open('empleados.json', 'r', encoding='utf-8') as f:
        datos = json.load(f)
    
    df = pd.DataFrame(datos['empleados'])
    
    df = df[df['calidad_candidato'].notna() & (df['calidad_candidato'] != '')]

    if df.empty:
        print("❌ ERROR: No se encontraron empleados calificados en 'empleados.json'.")
        print("Asegúrate de que el JSON tenga la columna 'calidad_candidato' con valores.")
        exit()

    print(f"✅ Se cargaron {len(df)} registros de ejemplo para entrenar.")
except Exception as e:
    print(f"❌ Error al cargar 'empleados.json': {e}")
    exit()

# 2. Preprocesamiento de Datos (Convertir texto a números)
df.fillna('Ninguna', inplace=True)
df['experiencia_limpia'] = df['experiencia'].apply(limpiar_experiencia)

X = pd.get_dummies(df[['experiencia_limpia', 'licencias']], drop_first=True)
y = df['calidad_candidato']

joblib.dump(X.columns, 'columnas_modelo.pkl')
print("Columnas del modelo guardadas.")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'modelo_reclutador.pkl')
print("🧠 ¡Entrenamiento completado! El modelo 'modelo_reclutador.pkl' ha sido guardado.")
