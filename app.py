import pandas as pd
import joblib
import json
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def limpiar_experiencia(texto):
    if isinstance(texto, str):
        if 'sin exper' in texto.lower(): return 'Sin Experiencia'
        if 'basic' in texto.lower(): return 'Basica'
        if 'intermedia' in texto.lower(): return 'Intermedia'
        if 'experta' in texto.lower(): return 'Experta'
    return 'Ninguna'



app = Flask(__name__)
app.config['SECRET_KEY'] = 'tu_clave_secreta_aqui'


class GestionEmpleados:
    def __init__(self, archivo_json):
        self.archivo_json = archivo_json
        self.datos = self._cargar_datos()
        try:
            self.model = joblib.load('modelo_reclutador.pkl')
            self.model_columns = joblib.load('columnas_modelo.pkl')
            print("🤖 Modelo de IA cargado correctamente.")
        except FileNotFoundError:
            print("⚠️ ADVERTENCIA: No se encontraron los archivos del modelo de IA.")
            self.model = None

    def _cargar_datos(self):
        if not os.path.exists(self.archivo_json): return {"empresa": "VMC SEGUCLEAN S.A de C.V", "empleados": []}
        try:
            with open(self.archivo_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error al cargar {self.archivo_json}: {e}")
            return {"empresa": "VMC SEGUCLEAN S.A de C.V", "empleados": []}

    def _guardar_datos(self):
        try:
            with open(self.archivo_json, 'w', encoding='utf-8') as f:
                json.dump(self.datos, f, indent=2, ensure_ascii=False)
            print("💾 ¡Datos guardados en el JSON!")
        except Exception as e:
            print(f"❌ Error al guardar en {self.archivo_json}: {e}")

    def _generar_nuevo_id(self):
        if not self.datos['empleados']: return 1
        return max(emp['id'] for emp in self.datos['empleados']) + 1

    def obtener_todos(self):
        return self.datos['empleados']

    def obtener_por_id(self, id_empleado):
        for emp in self.datos['empleados']:
            if emp['id'] == id_empleado:
                return emp
        return None 

    def agregar_empleado(self, datos_formulario):
        try:
            experiencia_input = datos_formulario.get('experiencia')
            licencias_input = datos_formulario.get('licencias')
            calidad_predicha = None
            if self.model:
                calidad_predicha = self.predecir_calidad(experiencia_input, licencias_input)
            
            nuevo_empleado = {
                "id": self._generar_nuevo_id(),
                "nombre": datos_formulario.get('nombre'),
                "apellido": datos_formulario.get('apellido') or None,
                "ubicacion": datos_formulario.get('ubicacion') or None,
                "experiencia": experiencia_input,
                "licencias": licencias_input or None,
                "estado": datos_formulario.get('estado') or None,
                "turno": datos_formulario.get('turno') or None,
                "fecha_contratacion": datos_formulario.get('fecha_contratacion') or None,
                "calidad_candidato": calidad_predicha
            }
            self.datos['empleados'].append(nuevo_empleado)
            self._guardar_datos()
            return True, calidad_predicha
        except Exception as e:
            print(f"Error al agregar empleado: {e}")
            return False, None

    def actualizar_empleado(self, id_empleado, datos_formulario):
        empleado = self.obtener_por_id(id_empleado)
        if not empleado:
            return False

        empleado['nombre'] = datos_formulario.get('nombre')
        empleado['apellido'] = datos_formulario.get('apellido') or None
        empleado['ubicacion'] = datos_formulario.get('ubicacion') or None
        empleado['experiencia'] = datos_formulario.get('experiencia')
        empleado['licencias'] = datos_formulario.get('licencias') or None
        empleado['estado'] = datos_formulario.get('estado') or None
        empleado['turno'] = datos_formulario.get('turno') or None
        empleado['fecha_contratacion'] = datos_formulario.get('fecha_contratacion') or None
        
        self._guardar_datos()
        return True

    def eliminar_empleado(self, id_empleado):
        empleado_encontrado = self.obtener_por_id(id_empleado)
        if empleado_encontrado:
            self.datos['empleados'].remove(empleado_encontrado)
            self._guardar_datos()
            return True
        return False

    def predecir_calidad(self, experiencia, licencias):
        try:
            experiencia_limpia = limpiar_experiencia(experiencia)
            data = {'experiencia_limpia': [experiencia_limpia], 'licencias': [licencias]}
            df_nuevo = pd.DataFrame(data)
            df_nuevo.fillna('Ninguna', inplace=True)
            df_nuevo_procesado = pd.get_dummies(df_nuevo)
            df_final = df_nuevo_procesado.reindex(columns=self.model_columns, fill_value=0)
            prediccion = self.model.predict(df_final)
            return prediccion[0]
        except Exception as e:
            print(f"Error durante la predicción: {e}")
            return "Indeterminado"

gestor = GestionEmpleados('empleados.json')


@app.route('/')
def index():
    empleados = gestor.obtener_todos()
    return render_template('index.html', empleados=empleados)

@app.route('/agregar', methods=['GET', 'POST'])
def agregar():
    if request.method == 'POST':
        exito, prediccion = gestor.agregar_empleado(request.form)
        if exito:
            flash(f"¡Empleado '{request.form.get('nombre')}' agregado! Clasificación IA: {prediccion}", 'success')
        else:
            flash("Hubo un error al agregar el empleado.", 'danger')
        return redirect(url_for('index'))
    return render_template('agregar.html')

@app.route('/editar/<int:id_empleado>', methods=['GET', 'POST'])
def editar(id_empleado):
    empleado = gestor.obtener_por_id(id_empleado)
    if not empleado:
        flash("Empleado no encontrado.", 'danger')
        return redirect(url_for('index'))

    if request.method == 'POST':
        exito = gestor.actualizar_empleado(id_empleado, request.form)
        if exito:
            flash(f"Empleado '{request.form.get('nombre')}' actualizado correctamente.", 'success')
        else:
            flash("Hubo un error al actualizar el empleado.", 'danger')
        return redirect(url_for('index'))

    return render_template('editar.html', empleado=empleado)

@app.route('/eliminar/<int:id_empleado>')
def eliminar(id_empleado):
    exito = gestor.eliminar_empleado(id_empleado)
    if exito:
        flash(f"Empleado con ID {id_empleado} eliminado.", 'success')
    else:
        flash(f"No se pudo encontrar o eliminar al empleado con ID {id_empleado}.", 'danger')
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)