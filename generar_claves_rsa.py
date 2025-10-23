from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

print("Generando par de claves RSA...")

private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
public_key = private_key.public_key()

pem_private = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)
with open('private_key.pem', 'wb') as f: f.write(pem_private)
print("🔑 Clave privada guardada en 'private_key.pem'")

pem_public = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)
with open('public_key.pem', 'wb') as f: f.write(pem_public)
print("🔑 Clave pública guardada en 'public_key.pem'")

print("\n¡IMPORTANTE! Copia el contenido de 'public_key.pem' para pegarlo en login.html.")