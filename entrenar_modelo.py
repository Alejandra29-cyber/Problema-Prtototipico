import pandas as pd
import json
import os
try:
    # Importaciones necesarias para la Red Neuronal
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder # Para escalar y codificar
    import joblib
except ImportError:
    print("Error: Faltan librerías. Ejecuta: pip install scikit-learn pandas joblib")
    exit()

print("🤖 Iniciando el entrenamiento de la RED NEURONAL...")

# Función para limpiar experiencia (sin cambios)
def limpiar_experiencia(texto):
    if isinstance(texto, str):
        if 'sin exper' in texto.lower(): return 'Sin Experiencia'
        if 'basic' in texto.lower(): return 'Basica'
        if 'intermedia' in texto.lower(): return 'Intermedia'
        if 'experta' in texto.lower(): return 'Experta'
    return 'Ninguna'

# --- Rutas a archivos (Definimos nombres nuevos para los archivos del modelo NN) ---
carpeta_datos = 'Data'
ruta_empleados_json = os.path.join(carpeta_datos, 'empleados.json')
ruta_modelo_nn_pkl = os.path.join(carpeta_datos, 'modelo_red_neuronal.pkl') # Modelo NN
ruta_scaler_pkl = os.path.join(carpeta_datos, 'scaler.pkl')                 # Escaldor
ruta_encoder_pkl = os.path.join(carpeta_datos, 'label_encoder.pkl')        # Codificador Etiquetas
ruta_columnas_pkl = os.path.join(carpeta_datos, 'columnas_modelo_nn.pkl')   # Columnas usadas

# Asegurarse de que la carpeta Data exista
if not os.path.exists(carpeta_datos):
    print(f"Creando carpeta '{carpeta_datos}'...")
    os.makedirs(carpeta_datos)

# 1. Carga de Datos (Sin cambios, solo usa la ruta definida)
try:
    with open(ruta_empleados_json, 'r', encoding='utf-8') as f: datos = json.load(f)
    df = pd.DataFrame(datos['empleados'])
    # Filtramos solo registros con calificación válida
    df = df[df['calidad_candidato'].notna() & (df['calidad_candidato'] != '')]
    if df.empty:
        print(f"❌ ERROR: No se encontraron empleados calificados con 'calidad_candidato' en '{ruta_empleados_json}'.")
        print("   Asegúrate de que el JSON tenga ejemplos calificados.")
        exit()
    print(f"✅ Se cargaron {len(df)} registros de ejemplo.")
except FileNotFoundError:
     print(f"❌ ERROR: No se encontró el archivo '{ruta_empleados_json}'.")
     exit()
except Exception as e: print(f"❌ Error al cargar '{ruta_empleados_json}': {e}"); exit()

# 2. Preprocesamiento de Datos
df.fillna('Ninguna', inplace=True)
df['experiencia_limpia'] = df['experiencia'].apply(limpiar_experiencia)

# Paso A: One-Hot Encoding
X_dummies = pd.get_dummies(df[['experiencia_limpia', 'licencias']], drop_first=True)
# --- GUARDA las columnas en la carpeta Data ---
try:
    joblib.dump(X_dummies.columns, ruta_columnas_pkl)
    print(f"Columnas del modelo guardadas en '{ruta_columnas_pkl}'.")
except Exception as e: print(f"❌ Error al guardar '{ruta_columnas_pkl}': {e}"); exit()

# PASO B: Escalar Datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dummies)
# --- GUARDA el escalador en la carpeta Data ---
try:
    joblib.dump(scaler, ruta_scaler_pkl)
    print(f"Escalador de datos guardado en '{ruta_scaler_pkl}'.")
except Exception as e: print(f"❌ Error al guardar '{ruta_scaler_pkl}': {e}"); exit()


# PASO C: Codificar Etiquetas
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(df['calidad_candidato'])
# --- GUARDA el codificador en la carpeta Data ---
try:
    joblib.dump(encoder, ruta_encoder_pkl)
    print(f"Codificador de etiquetas guardado en '{ruta_encoder_pkl}'.")
    print(f"   Etiquetas codificadas: {list(encoder.classes_)}")
except Exception as e: print(f"❌ Error al guardar '{ruta_encoder_pkl}': {e}"); exit()


# 3. Entrenamiento de la Red Neuronal
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

model_nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, activation='relu', solver='adam', early_stopping=True)
print("Entrenando la red neuronal...")
model_nn.fit(X_train, y_train)

# 4. Evaluación (Opcional)
accuracy = model_nn.score(X_test, y_test)
print(f"Precisión de la Red Neuronal: {accuracy * 100:.2f}%")

# --- GUARDA el modelo entrenado en la carpeta Data ---
try:
    joblib.dump(model_nn, ruta_modelo_nn_pkl)
    print(f"🧠 ¡Entrenamiento completado! Modelo guardado en '{ruta_modelo_nn_pkl}'.")
except Exception as e:
    print(f"❌ Error al guardar '{ruta_modelo_nn_pkl}': {e}")
    exit()

