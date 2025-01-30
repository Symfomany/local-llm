import os
from openai import OpenAI


print(os.environ.get("OPENAI_KEY"))
# Initialize the OpenAI client with your API key
oai_client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))

# Create a chat completion request
completion = oai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Tu es un dev Python chez Enedis."},
        {"role": "user", "content": "Ecris une fonction factorielle en Python."},
    ],
)

# Print the response from the model
response_message = completion.choices[0].message.content

# Print the response message
print(response_message)
