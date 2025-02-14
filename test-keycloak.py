import jwt
import time
import requests

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


print("signed_jwt...", signed_jwt)
# signed_jwt = "MIIClzCCAX8CBgGU8bb0yTANBgkqhkiG9w0BAQsFADAPMQ0wCwYDVQQDDARqdWp1MB4XDTI1MDIxMDIxMTMwNloXDTM1MDIxMDIxMTQ0NlowDzENMAsGA1UEAwwEanVqdTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAPkrLim8YDGgJ6IQ+gaer/coX4R3M3WUSF5EMmb14bLipqwzaZ7Nd8ACxvo9F7dds+duZVEYKewWkT+mYutpv3e41rXQEAIG03a89xqI+yvCIoRXlOAx6q3loeguSxHCzz4RaV/7/66qf3YnTp/Q6Q1MNskWW0vSpf+nI97EYiAY7Z77MVz96L6/LpRN+9BRnLzcft2lY/h/6eKwDdJCwo+mZ6oVKDHa2Cb5HAShCJBwcymbjrNr98BwYNOsOd41LZjx6+0ERQ8mLKR73eUg1VYiSUWvu1+SEpT8k3ShK2BXuHbbyXJNEtvpCqbIFY54uQJb+s9mRy3gTq9OxNfYnAcCAwEAATANBgkqhkiG9w0BAQsFAAOCAQEAjneP300BWMYPxYZmT1rNFTLZZUqteDRnWg2p5Btsxcp/esb62c3pImuuJO2mNvvvOPRbsGmScQVP5kJQykvZPs9uIz25Hg/8JZvveadqKnUXIGLiK/Ufqe1WHbMl7o9HWpjlibs0DQEPO3hsIDm3xbU2M/X2Bst6ZUh2Dno5iDkZy18+TJRO0hqov67oncsn5Y7aJYBO2eHloYLsMRZFqa4+N/a7q0fgimjW76yb3Wl9rZBy2u2hbCQB3NPS39knDm942wDhlYxR14W+zkT0frWlVTbKy99mtf1lsXvpfN5f7LxtwZ4J40UOi++7LErC0+sUUY9JNUz1LGPBaBmGsA=="
# Effectuer la requête POST pour obtenir un jeton d'accès
url = f"http://localhost:9080/realms/{realm}/protocol/openid-connect/token"
data = {
    "grant_type": "client_credentials",
    "client_id": client_id,
    "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
    "client_assertion": signed_jwt,
}

response = requests.post(url, data=data)

# Afficher la réponse
if response.status_code == 200:
    print("Authentification réussie. Jeton d'accès :")
    print(response.json())
else:
    print("Erreur lors de l'authentification.")
    print("Statut HTTP :", response.status_code)
    print("Réponse :", response.text)
