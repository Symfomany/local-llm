FROM python:3.9

WORKDIR /app

COPY myapp.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "myapp.py"]
