
FROM python:3.11-slim

# ⚡ NEW: Define a build argument that will hold the key ⚡
ARG BLOB_STORAGE_KEY

# ⚡ NEW: Set the environment variable inside the container using the build argument ⚡
ENV BLOB_STORAGE_KEY=$BLOB_STORAGE_KEY

WORKDIR /home/site/wwwroot
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /home/site/wwwroot

EXPOSE 80


CMD ["gunicorn", "web_app.app:app", "--bind", "0.0.0.0:80"]