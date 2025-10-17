import mysql.connector
from mysql.connector import Error
import pandas as pd
import joblib

# Función para limpiar y estandarizar la experiencia (LA MISMA que en el entrenador)
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

class GestionEmpleados:
    def __init__(self):
        try:
            self.connection = mysql.connector.connect(
                host='localhost', user='root', password='', database='empresa_db', buffered=True
            )
            if self.connection.is_connected():
                print("✅ ¡Conexión exitosa a la base de datos MySQL!")
        except Error as e:
            print(f"❌ Error al conectar a la base de datos: {e}")
            self.connection = None

        try:
            self.model = joblib.load('modelo_reclutador.pkl')
            self.model_columns = joblib.load('columnas_modelo.pkl')
            print("🤖 Modelo de IA cargado correctamente.")
        except FileNotFoundError:
            print("⚠️ Advertencia: No se encontraron los archivos del modelo.")
            print("⚠️ La predicción de candidatos no estará disponible.")
            print("⚠️ Ejecuta el script 'entrenar_modelo.py' para crearlos.")
            self.model = None

    def _ejecutar_query(self, query, params=None, fetch=False):
        if not self.connection or not self.connection.is_connected():
            print("❌ No hay conexión a la base de datos.")
            return None
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(query, params or ())
            if fetch: return cursor.fetchall()
            else: self.connection.commit(); print("💾 ¡Operación completada!")
        except Error as e:
            print(f"❌ Error al ejecutar la consulta: {e}")
            return None
        finally:
            cursor.close()

    def mostrar_empleados(self):
        print("\n--- LISTA DE EMPLEADOS ---")
        query = "SELECT id, nombre, apellido, ubicacion, experiencia, licencias, estado, turno, fecha_contratacion, calidad_candidato FROM empleados"
        resultados = self._ejecutar_query(query, fetch=True)
        if resultados:
            df = pd.DataFrame(resultados)
            df.fillna('-', inplace=True)
            print(df.to_string(index=False))
        else:
            print("No hay empleados para mostrar.")

    def agregar_empleado(self):
        print("\n--- AGREGAR NUEVO EMPLEADO (CON ANÁLISIS IA) ---")
        try:
            nombre = input("Nombre: ")
            apellido = input("Apellido: ")
            ubicacion = input("Ubicación (Servicio): ")
            estado = input("Estado (Activo, Inactivo): ")
            turno = input("Turno (24x24, 12x12): ")
            fecha = input("Fecha de contratación (YYYY-MM-DD): ")
            
            experiencia_input = input("Experiencia (Basica, Intermedia, Experta, Sin experincia): ")
            licencias_input = input("Licencias (separadas por coma, ej: 'Defensa personal'): ")
            
            calidad_predicha = None
            if self.model:
                calidad_predicha = self.predecir_calidad(experiencia_input, licencias_input)
                print(f"💡 Análisis de IA: Este candidato es clasificado como: '{calidad_predicha}'")
            else:
                print("Saltando análisis de IA (modelo no cargado).")

            params = (
                nombre, 
                apellido if apellido else None, 
                ubicacion if ubicacion else None, 
                experiencia_input, # Guardamos el texto original
                licencias_input if licencias_input else None, 
                estado if estado else None, 
                turno if turno else None, 
                fecha if fecha else None, 
                calidad_predicha
            )
            
            query = """
            INSERT INTO empleados (nombre, apellido, ubicacion, experiencia, licencias, estado, turno, fecha_contratacion, calidad_candidato)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            self._ejecutar_query(query, params)
            
        except Exception as e:
            print(f"Error en la entrada de datos: {e}")

    def actualizar_empleado(self):
        print("\n--- ACTUALIZAR EMPLEADO ---")
        try:
            id_empleado = int(input("Ingrese el ID del empleado a actualizar: "))
            campos_permitidos = ['nombre', 'apellido', 'ubicacion', 'experiencia', 'licencias', 'estado', 'turno', 'fecha_contratacion', 'calidad_candidato']
            campo_str = ", ".join(campos_permitidos)
            campo = input(f"¿Qué campo desea actualizar? ({campo_str}): ").lower()
            
            if campo not in campos_permitidos:
                print("❌ Nombre de campo no válido.")
                return

            nuevo_valor = input(f"Ingrese el nuevo valor para '{campo}' (dejar en blanco para borrarlo): ")
            params = (nuevo_valor if nuevo_valor else None, id_empleado)
            query = f"UPDATE empleados SET {campo} = %s WHERE id = %s"
            self._ejecutar_query(query, params)
        except (ValueError, Error) as e:
            print(f"Error al actualizar: {e}")

    def eliminar_empleado(self):
        print("\n--- ELIMINAR EMPLEADO ---")
        try:
            id_empleado = int(input("Ingrese el ID del empleado a eliminar: "))
            confirmacion = input(f"❓ ¿Está seguro? (s/n): ").lower()
            if confirmacion == 's':
                self._ejecutar_query("DELETE FROM empleados WHERE id = %s", (id_empleado,))
            else:
                print("Operación cancelada.")
        except ValueError:
            print("❌ ID no válido. Debe ser un número.")

    def predecir_calidad(self, experiencia, licencias):
        """Usa el modelo cargado para predecir la calidad."""
        try:
            # --- ESTANDARIZAMOS LA ENTRADA ---
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
            return "Indeterminado" # Fallback
            
    def __del__(self):
        if hasattr(self, 'connection') and self.connection and self.connection.is_connected():
            self.connection.close()
            print("\nConexión a la base de datos cerrada.")

def menu_principal(gestor):
    if not gestor.connection:
        print("No se pudo iniciar el programa.")
        return
        
    while True:
        print("\n" + "="*50)
        print("   SISTEMA DE GGESTIÓN VMC SEGUCLEAN (CON IA)")
        print("="*50)
        print("1. Ver todos los empleados")
        print("2. Agregar nuevo empleado (con IA)")
        print("3. Actualizar empleado")
        print("4. Eliminar empleado")
        print("5. Salir")
        opcion = input("\nSeleccione una opción (1-5): ")
        if opcion == '1': gestor.mostrar_empleados()
        elif opcion == '2': gestor.agregar_empleado()
        elif opcion == '3': gestor.actualizar_empleado()
        elif opcion == '4': gestor.eliminar_empleado()
        elif opcion == '5':
            print("👋 ¡Hasta luego!")
            break
        else: print("❌ Opción no válida.")

if __name__ == "__main__":
    gestor = GestionEmpleados()
    menu_principal(gestor)