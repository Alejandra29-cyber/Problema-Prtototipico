import pandas as pd
import joblib
import json
import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
import base64
import traceback

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    print("Advertencia: Faltan librerías de IA.")

def limpiar_experiencia(texto):
    if isinstance(texto, str):
        if 'sin exper' in texto.lower(): return 'Sin Experiencia'
        if 'basic' in texto.lower(): return 'Basica'
        if 'intermedia' in texto.lower(): return 'Intermedia'
        if 'experta' in texto.lower(): return 'Experta'
    return 'Ninguna'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cambia_esto_por_algo_aleatorio_y_largo!'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "Por favor, inicia sesión."
login_manager.login_message_category = "info"

private_key = None
try:
    ruta_clave_privada = os.path.join('Data', 'private_key.pem')
    if not os.path.exists(ruta_clave_privada):
        ruta_clave_privada = 'private_key.pem'

    with open(ruta_clave_privada, "rb") as key_file:
        private_key = serialization.load_pem_private_key(key_file.read(), password=None)
    print(f"🔑 Clave privada RSA cargada desde '{ruta_clave_privada}'.")
except FileNotFoundError:
    print(f"❌ ERROR CRÍTICO: No se encontró 'private_key.pem' ni en la raíz ni en 'Data'. Ejecuta 'generar_claves_rsa.py'.")
except Exception as e:
    print(f"❌ Error al cargar la clave privada: {e}")

def rsa_decrypt(encrypted_data_b64):
    if not private_key:
        print("❌ Descifrado fallido: La clave privada RSA no se cargó correctamente.")
        return None
    try:
        encrypted_data = base64.b64decode(encrypted_data_b64)
        oaep_padding = padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA1()),
            algorithm=hashes.SHA256(),
            label=None
        )
        original_data_bytes = private_key.decrypt(encrypted_data, oaep_padding)
        original_string = original_data_bytes.decode('utf-8')
        return original_string
    except base64.binascii.Error as b64e:
        print(f"❌ Error al decodificar Base64: {b64e}")
        print(f"   Datos recibidos (primeros 50): {encrypted_data_b64[:50]}...")
        return None
    except ValueError as ve:
        print(f"❌ Error de descifrado (ValueError): {ve}")
        print("   Posibles causas: Clave pública/privada no coinciden, datos alterados, o padding incorrecto.")
        return None
    except TypeError as te:
        print(f"❌ TypeError durante el descifrado: {te}")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"❌ Error inesperado al descifrar: {type(e).__name__} - {e}")
        return None

class User(UserMixin):
    def __init__(self, id, username, password_hash): self.id = id; self.username = username; self.password_hash = password_hash

def cargar_usuarios():
    ruta_usuarios = os.path.join('Data', 'usuarios.json')
    try:
        with open(ruta_usuarios, 'r', encoding='utf-8') as f: return json.load(f)['usuarios']
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el archivo de usuarios en '{ruta_usuarios}'.")
        return []
    except Exception as e:
        print(f"Error al cargar {ruta_usuarios}: {e}"); return []

usuarios_db = cargar_usuarios()
users = {u['id']: User(u['id'], u['username'], u['password_hash']) for u in usuarios_db}

@login_manager.user_loader
def load_user(user_id): return users.get(int(user_id))

class GestionEmpleados:
    def __init__(self, archivo_json):
        self.archivo_json = archivo_json
        self.datos = self._cargar_datos()
        self.model = None
        self.model_columns = None
        try:
            ruta_modelo = os.path.join('Data', 'modelo_reclutador.pkl')
            ruta_columnas = os.path.join('Data', 'columnas_modelo.pkl')
            self.model = joblib.load(ruta_modelo)
            self.model_columns = joblib.load(ruta_columnas)
            print("🤖 Modelo IA cargado.")
        except FileNotFoundError: print("⚠️ Modelo IA no encontrado en 'Data'.")
        except Exception as e: print(f"Error al cargar modelo IA: {e}")

    def _cargar_datos(self):
        if not os.path.exists(self.archivo_json):
            print(f"Advertencia: Archivo '{self.archivo_json}' no encontrado. Se creará uno vacío.")
            return {"empresa": "VMC SEGUCLEAN S.A de C.V", "empleados": []}
        try:
            with open(self.archivo_json, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception as e: print(f"Error al cargar {self.archivo_json}: {e}"); return {"empresa": "VMC SEGUCLEAN S.A de C.V", "empleados": []}

    def _guardar_datos(self):
        try:
            with open(self.archivo_json, 'w', encoding='utf-8') as f: json.dump(self.datos, f, indent=2, ensure_ascii=False)
            print("💾 JSON guardado.")
        except Exception as e: print(f"❌ Error al guardar JSON: {e}")

    def _generar_nuevo_id(self):
        if not self.datos.get('empleados'): return 1
        ids_numericos = [emp.get('id', 0) for emp in self.datos['empleados'] if isinstance(emp.get('id'), int)]
        if not ids_numericos: return 1
        return max(ids_numericos) + 1

    def obtener_todos(self): return self.datos.get('empleados', [])

    def obtener_por_id(self, id_empleado):
        for emp in self.datos.get('empleados', []):
            if emp.get('id') == id_empleado: return emp
        return None

    def agregar_empleado(self, datos_formulario):
        try:
            experiencia_input = datos_formulario.get('experiencia'); licencias_input = datos_formulario.get('licencias'); calidad_predicha = "Indeterminado"
            if self.model: calidad_predicha = self.predecir_calidad(experiencia_input, licencias_input)
            nuevo_empleado = {"id": self._generar_nuevo_id(), "nombre": datos_formulario.get('nombre'),"apellido": datos_formulario.get('apellido') or None,"ubicacion": datos_formulario.get('ubicacion') or None,"experiencia": experiencia_input,"licencias": licencias_input or None,"estado": datos_formulario.get('estado') or None,"turno": datos_formulario.get('turno') or None,"fecha_contratacion": datos_formulario.get('fecha_contratacion') or None,"calidad_candidato": calidad_predicha}
            if 'empleados' not in self.datos: self.datos['empleados'] = []
            self.datos['empleados'].append(nuevo_empleado); self._guardar_datos(); return True, calidad_predicha
        except Exception as e: print(f"Error al agregar: {e}"); return False, None

    def actualizar_empleado(self, id_empleado, datos_formulario):
        empleado = self.obtener_por_id(id_empleado);
        if not empleado: return False
        empleado['nombre'] = datos_formulario.get('nombre'); empleado['apellido'] = datos_formulario.get('apellido') or None; empleado['ubicacion'] = datos_formulario.get('ubicacion') or None; empleado['experiencia'] = datos_formulario.get('experiencia'); empleado['licencias'] = datos_formulario.get('licencias') or None; empleado['estado'] = datos_formulario.get('estado') or None; empleado['turno'] = datos_formulario.get('turno') or None; empleado['fecha_contratacion'] = datos_formulario.get('fecha_contratacion') or None
        self._guardar_datos(); return True

    def eliminar_empleado(self, id_empleado):
        empleado_encontrado = self.obtener_por_id(id_empleado)
        if empleado_encontrado:
            if 'empleados' in self.datos:
                self.datos['empleados'].remove(empleado_encontrado)
                self._guardar_datos()
                return True
            else: print("Error: La lista de empleados no existe."); return False
        return False

    def predecir_calidad(self, experiencia, licencias):
        if not self.model: return "Indeterminado"
        try:
            experiencia_limpia = limpiar_experiencia(experiencia); data = {'experiencia_limpia': [experiencia_limpia], 'licencias': [licencias]}; df_nuevo = pd.DataFrame(data); df_nuevo.fillna('Ninguna', inplace=True); df_nuevo_procesado = pd.get_dummies(df_nuevo); df_final = df_nuevo_procesado.reindex(columns=self.model_columns, fill_value=0); prediccion = self.model.predict(df_final)
            return prediccion[0]
        except Exception as e: print(f"Error predicción: {e}"); return "Indeterminado"

ruta_empleados = os.path.join('Data', 'empleados.json')
gestor = GestionEmpleados(ruta_empleados)

@app.route('/')
@login_required
def index():
    empleados = gestor.obtener_todos(); return render_template('index.html', empleados=empleados)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password_cifrada_b64 = request.form['password_encrypted']
        password_descifrada = rsa_decrypt(password_cifrada_b64)
        if not password_descifrada:
            flash('Error al procesar contraseña.', 'danger'); return render_template('login.html')
        user_encontrado = None
        for u in usuarios_db:
            if u.get('username') == username:
                stored_hash = u.get('password_hash')
                if stored_hash and check_password_hash(stored_hash, password_descifrada):
                    user_encontrado = users.get(u.get('id')); break
        if user_encontrado:
            login_user(user_encontrado); flash('¡Inicio de sesión exitoso!', 'success')
            next_page = request.args.get('next'); return redirect(next_page or url_for('index'))
        else:
            flash('Usuario o contraseña incorrectos.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user(); flash('Has cerrado sesión.', 'info'); return redirect(url_for('login'))

@app.route('/agregar', methods=['GET', 'POST'])
@login_required
def agregar():
    if request.method == 'POST':
        exito, prediccion = gestor.agregar_empleado(request.form)
        if exito: flash(f"Empleado '{request.form.get('nombre')}' agregado! IA: {prediccion}", 'success')
        else: flash("Error al agregar.", 'danger')
        return redirect(url_for('index'))
    return render_template('agregar.html')

@app.route('/editar/<int:id_empleado>', methods=['GET', 'POST'])
@login_required
def editar(id_empleado):
    empleado = gestor.obtener_por_id(id_empleado)
    if not empleado: flash("Empleado no encontrado.", 'danger'); return redirect(url_for('index'))
    if request.method == 'POST':
        exito = gestor.actualizar_empleado(id_empleado, request.form)
        if exito: flash(f"Empleado '{request.form.get('nombre')}' actualizado.", 'success')
        else: flash("Error al actualizar.", 'danger')
        return redirect(url_for('index'))
    return render_template('editar.html', empleado=empleado)

@app.route('/eliminar/<int:id_empleado>')
@login_required
def eliminar(id_empleado):
    exito = gestor.eliminar_empleado(id_empleado)
    if exito: flash(f"Empleado ID {id_empleado} eliminado.", 'success')
    else: flash(f"Error al eliminar ID {id_empleado}.", 'danger')
    return redirect(url_for('index'))

if __name__ == "__main__":
    if private_key: app.run(debug=True)
    else: print("El servidor no puede iniciar sin la clave privada RSA.")

