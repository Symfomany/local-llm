
import jwt
import time

# Charger la clé privée
with open("private.pem", "r") as key_file:
    private_key = key_file.read()

# Informations pour le JWT
client_id = "juju"
realm = "example-realm"
audience = f"http://localhost:9080/realms/{realm}"

# Générer le JWT signé
payload = {
    "iss": client_id,
    "sub": client_id,
    "aud": audience,
    "iat": int(time.time()),
    "exp": int(time.time()) + 300,  # Expiration dans 5 minutes
}
signed_jwt = jwt.encode(payload, private_key, algorithm="ES256")

print(f"JWT signé : {signed_jwt}")
