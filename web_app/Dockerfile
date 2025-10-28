
FROM python:3.11-slim

ENV PYTHONUNBUFFERED 1
ENV PORT 80 

WORKDIR /home/site/wwwroot

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /home/site/wwwroot

EXPOSE 80


CMD ["gunicorn", "web_app.app:app", "--bind", "0.0.0.0:80"]