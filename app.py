import os
import json
import base64
import traceback
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user, login_required, current_user
)
from werkzeug.security import check_password_hash
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    print("Advertencia: Faltan librerías de IA.")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cambia_esto_por_algo_aleatorio_y_largo!'

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = "Por favor, inicia sesión."
login_manager.login_message_category = "info"


def limpiar_experiencia(texto):
    if isinstance(texto, str):
        texto = texto.lower()
        if 'sin exper' in texto: return 'Sin Experiencia'
        if 'basic' in texto: return 'Basica'
        if 'intermedia' in texto: return 'Intermedia'
        if 'experta' in texto: return 'Experta'
    return 'Ninguna'


def cargar_clave_privada():
    posibles_rutas = [
        os.path.join('Data', 'private_key.pem'),
        'private_key.pem'
    ]
    for ruta in posibles_rutas:
        if os.path.exists(ruta):
            with open(ruta, "rb") as key_file:
                clave = serialization.load_pem_private_key(key_file.read(), password=None)
            print(f"🔑 Clave privada RSA cargada desde '{ruta}'.")
            return clave
    print("❌ ERROR CRÍTICO: No se encontró 'private_key.pem'. Ejecuta 'generar_claves_rsa.py'.")
    return None


private_key = cargar_clave_privada()


def rsa_decrypt(encrypted_data_b64):
    if not private_key:
        print("❌ Descifrado fallido: La clave privada RSA no se cargó correctamente.")
        return None
    try:
        encrypted_data = base64.b64decode(encrypted_data_b64)
        original_bytes = private_key.decrypt(encrypted_data, padding.PKCS1v15())
        return original_bytes.decode('utf-8')
    except Exception as e:
        print(f"❌ Error al descifrar: {type(e).__name__} - {e}")
        traceback.print_exc()
        return None


class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash


def cargar_usuarios():
    ruta = os.path.join('Data', 'usuarios.json')
    if not os.path.exists(ruta):
        print(f"❌ ERROR: No se encontró '{ruta}'.")
        return []
    try:
        with open(ruta, 'r', encoding='utf-8') as f:
            return json.load(f)['usuarios']
    except Exception as e:
        print(f"Error al cargar usuarios: {e}")
        return []


usuarios_db = cargar_usuarios()
users = {u['id']: User(u['id'], u['username'], u['password_hash']) for u in usuarios_db}


@login_manager.user_loader
def load_user(user_id):
    return users.get(int(user_id))


class GestionEmpleados:
    def __init__(self, archivo_json):
        self.archivo_json = archivo_json
        self.datos = self._cargar_datos()
        self.model = None
        self.model_columns = None
        self._cargar_modelo()

    def _cargar_modelo(self):
        try:
            ruta_modelo = os.path.join('Data', 'modelo_reclutador.pkl')
            ruta_columnas = os.path.join('Data', 'columnas_modelo.pkl')
            self.model = joblib.load(ruta_modelo)
            self.model_columns = joblib.load(ruta_columnas)
            print("🤖 Modelo IA cargado.")
        except FileNotFoundError:
            print("⚠️ Modelo IA no encontrado.")
        except Exception as e:
            print(f"Error al cargar modelo IA: {e}")

    def _cargar_datos(self):
        if not os.path.exists(self.archivo_json):
            print(f"Archivo '{self.archivo_json}' no encontrado. Se creará vacío.")
            return {"empresa": "VMC SEGUCLEAN S.A de C.V", "empleados": []}
        try:
            with open(self.archivo_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error al cargar JSON: {e}")
            return {"empresa": "VMC SEGUCLEAN S.A de C.V", "empleados": []}

    def _guardar_datos(self):
        try:
            with open(self.archivo_json, 'w', encoding='utf-8') as f:
                json.dump(self.datos, f, indent=2, ensure_ascii=False)
            print("💾 JSON guardado.")
        except Exception as e:
            print(f"❌ Error al guardar JSON: {e}")

    def _generar_nuevo_id(self):
        empleados = self.datos.get('empleados', [])
        ids = [emp.get('id', 0) for emp in empleados if isinstance(emp.get('id'), int)]
        return max(ids, default=0) + 1

    def obtener_todos(self):
        return self.datos.get('empleados', [])

    def obtener_por_id(self, id_empleado):
        return next((e for e in self.datos.get('empleados', []) if e.get('id') == id_empleado), None)

    def agregar_empleado(self, datos):
        try:
            experiencia = datos.get('experiencia')
            licencias = datos.get('licencias')
            calidad = self.predecir_calidad(experiencia, licencias) if self.model else "Indeterminado"

            nuevo = {
                "id": self._generar_nuevo_id(),
                "nombre": datos.get('nombre'),
                "apellido": datos.get('apellido') or None,
                "ubicacion": datos.get('ubicacion') or None,
                "experiencia": experiencia,
                "licencias": licencias or None,
                "estado": datos.get('estado') or None,
                "turno": datos.get('turno') or None,
                "fecha_contratacion": datos.get('fecha_contratacion') or None,
                "calidad_candidato": calidad
            }

            self.datos.setdefault('empleados', []).append(nuevo)
            self._guardar_datos()
            return True, calidad
        except Exception as e:
            print(f"Error al agregar: {e}")
            return False, None

    def actualizar_empleado(self, id_empleado, datos):
        empleado = self.obtener_por_id(id_empleado)
        if not empleado:
            return False

        campos = ["nombre", "apellido", "ubicacion", "experiencia", "licencias", "estado", "turno", "fecha_contratacion"]
        for campo in campos:
            empleado[campo] = datos.get(campo) or None

        self._guardar_datos()
        return True

    def eliminar_empleado(self, id_empleado):
        empleado = self.obtener_por_id(id_empleado)
        if not empleado:
            return False
        self.datos['empleados'].remove(empleado)
        self._guardar_datos()
        return True

    def predecir_calidad(self, experiencia, licencias):
        if not self.model:
            return "Indeterminado"
        try:
            exp_limpia = limpiar_experiencia(experiencia)
            df = pd.DataFrame({'experiencia_limpia': [exp_limpia], 'licencias': [licencias]})
            df.fillna('Ninguna', inplace=True)
            df = pd.get_dummies(df).reindex(columns=self.model_columns, fill_value=0)
            return self.model.predict(df)[0]
        except Exception as e:
            print(f"Error en predicción: {e}")
            return "Indeterminado"


gestor = GestionEmpleados(os.path.join('Data', 'empleados.json'))


@app.route('/')
@login_required
def index():
    return render_template('index.html', empleados=gestor.obtener_todos())


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password_cifrada = request.form['password_encrypted']
        password = rsa_decrypt(password_cifrada)

        if not password:
            flash('Error al procesar contraseña.', 'danger')
            return render_template('login.html')

        for u in usuarios_db:
            if u['username'] == username and check_password_hash(u['password_hash'], password):
                login_user(users[u['id']])
                flash('¡Inicio de sesión exitoso!', 'success')
                return redirect(request.args.get('next') or url_for('index'))

        flash('Usuario o contraseña incorrectos.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Has cerrado sesión.', 'info')
    return redirect(url_for('login'))


@app.route('/agregar', methods=['GET', 'POST'])
@login_required
def agregar():
    if request.method == 'POST':
        exito, prediccion = gestor.agregar_empleado(request.form)
        flash(
            f"Empleado '{request.form.get('nombre')}' agregado. IA: {prediccion}" if exito else "Error al agregar.",
            'success' if exito else 'danger'
        )
        return redirect(url_for('index'))
    return render_template('agregar.html')


@app.route('/editar/<int:id_empleado>', methods=['GET', 'POST'])
@login_required
def editar(id_empleado):
    empleado = gestor.obtener_por_id(id_empleado)
    if not empleado:
        flash("Empleado no encontrado.", 'danger')
        return redirect(url_for('index'))

    if request.method == 'POST':
        exito = gestor.actualizar_empleado(id_empleado, request.form)
        flash(
            f"Empleado '{request.form.get('nombre')}' actualizado." if exito else "Error al actualizar.",
            'success' if exito else 'danger'
        )
        return redirect(url_for('index'))

    return render_template('editar.html', empleado=empleado)


@app.route('/eliminar/<int:id_empleado>')
@login_required
def eliminar(id_empleado):
    exito = gestor.eliminar_empleado(id_empleado)
    flash(
        f"Empleado ID {id_empleado} eliminado." if exito else f"Error al eliminar ID {id_empleado}.",
        'success' if exito else 'danger'
    )
    return redirect(url_for('index'))


if __name__ == "__main__":
    if private_key:
        app.run(debug=True)
    else:
        print("El servidor no puede iniciar sin la clave privada RSA.")
