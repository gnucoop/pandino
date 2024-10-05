FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN apt-get update
RUN apt-get -y install libpq-dev gcc
RUN pip install --upgrade pip
RUN pip3 install --prefer-binary -r requirements.txt
EXPOSE 5000
CMD ["flask","--app","main.py","run","--host=0.0.0.0"]
