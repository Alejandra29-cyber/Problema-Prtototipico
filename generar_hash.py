from werkzeug.security import generate_password_hash

password_a_hashear = input("Escribe la contraseña para el usuario: ")
hash_generado = generate_password_hash(password_a_hashear)
print("\nCopia este hash y pégalo en usuarios.json:")
print(hash_generado)