import pandas as pd
import json
import os # Necesario para manejar rutas de carpetas
try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib
except ImportError:
    print("Error: Faltan librerías. Ejecuta: pip install scikit-learn pandas joblib")
    exit()

print("🤖 Iniciando el entrenamiento del modelo desde el archivo JSON...")

# Función para limpiar y estandarizar la experiencia (sin cambios)
def limpiar_experiencia(texto):
    if isinstance(texto, str):
        if 'sin exper' in texto.lower(): return 'Sin Experiencia'
        if 'basic' in texto.lower(): return 'Basica'
        if 'intermedia' in texto.lower(): return 'Intermedia'
        if 'experta' in texto.lower(): return 'Experta'
    return 'Ninguna'

# --- DEFINIR RUTA A LA CARPETA DE DATOS ---
carpeta_datos = 'Data'
ruta_empleados_json = os.path.join(carpeta_datos, 'empleados.json')
ruta_columnas_pkl = os.path.join(carpeta_datos, 'columnas_modelo.pkl')
ruta_modelo_pkl = os.path.join(carpeta_datos, 'modelo_reclutador.pkl')

# Asegurarse de que la carpeta Data exista, si no, crearla
if not os.path.exists(carpeta_datos):
    print(f"Creando carpeta '{carpeta_datos}'...")
    os.makedirs(carpeta_datos)

# 1. Carga de Datos desde JSON
try:
    with open(ruta_empleados_json, 'r', encoding='utf-8') as f:
        datos = json.load(f)
    
    df = pd.DataFrame(datos['empleados'])
    df = df[df['calidad_candidato'].notna() & (df['calidad_candidato'] != '')]

    if df.empty:
        print(f"❌ ERROR: No se encontraron empleados calificados en '{ruta_empleados_json}'.")
        exit()
    print(f"✅ Se cargaron {len(df)} registros de ejemplo para entrenar.")
except FileNotFoundError:
     print(f"❌ ERROR: No se encontró el archivo '{ruta_empleados_json}'.")
     exit()
except Exception as e:
    print(f"❌ Error al cargar '{ruta_empleados_json}': {e}")
    exit()

# 2. Preprocesamiento de Datos
df.fillna('Ninguna', inplace=True)
df['experiencia_limpia'] = df['experiencia'].apply(limpiar_experiencia)
X = pd.get_dummies(df[['experiencia_limpia', 'licencias']], drop_first=True)
y = df['calidad_candidato']

# --- GUARDA las columnas en la carpeta Data ---
try:
    joblib.dump(X.columns, ruta_columnas_pkl)
    print(f"Columnas del modelo guardadas en '{ruta_columnas_pkl}'.")
except Exception as e:
    print(f"❌ Error al guardar '{ruta_columnas_pkl}': {e}")
    exit()


# 3. Entrenamiento del Modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --- GUARDA el modelo entrenado en la carpeta Data ---
try:
    joblib.dump(model, ruta_modelo_pkl)
    print(f"🧠 ¡Entrenamiento completado! Modelo guardado en '{ruta_modelo_pkl}'.")
except Exception as e:
    print(f"❌ Error al guardar '{ruta_modelo_pkl}': {e}")
    exit()
