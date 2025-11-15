FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN apt-get update
RUN apt-get -y install libpq-dev gcc
RUN pip install --upgrade pip
RUN pip3 install --prefer-binary -r requirements.txt
EXPOSE 5000
CMD ["python","--app","main.py","run","--host=0.0.0.0"]
#CMD ["gunicorn", "main:app", "-k", "gevent", "--workers", "1", "--worker-connections", "10", "--timeout", "300", "--bind", "0.0.0.0:5000"]
