import pandas as pd
import mysql.connector
from sklearn.ensemble import RandomForestClassifier
import joblib

print("🤖 Iniciando el entrenamiento del modelo...")

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

# 1. Conexión y Carga de Datos
try:
    connection = mysql.connector.connect(
        host='localhost', user='root', password='', database='empresa_db', buffered=True
    )
    # Cargamos SOLAMENTE los empleados que calificamos manualmente
    query = "SELECT experiencia, licencias, calidad_candidato FROM empleados WHERE calidad_candidato IS NOT NULL AND calidad_candidato != ''"
    df = pd.read_sql(query, connection)
    connection.close()

    if df.empty:
        print("❌ ERROR: No se encontraron empleados calificados en la base de datos.")
        print("Por favor, ejecuta el script SQL de calificación en phpMyAdmin primero.")
        exit()

    print(f"✅ Se cargaron {len(df)} registros de ejemplo para entrenar.")
except Exception as e:
    print(f"❌ Error al cargar datos: {e}")
    exit()

df.fillna('Ninguna', inplace=True)
df['experiencia_limpia'] = df['experiencia'].apply(limpiar_experiencia)

X = pd.get_dummies(df[['experiencia_limpia', 'licencias']], drop_first=True)
y = df['calidad_candidato']

joblib.dump(X.columns, 'columnas_modelo.pkl')
print("Columnas del modelo guardadas.")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'modelo_reclutador.pkl')
print("🧠 ¡Entrenamiento completado! El modelo 'modelo_reclutador.pkl' ha sido guardado.")