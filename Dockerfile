FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip3 install --prefer-binary -r requirements.txt
EXPOSE 5000
CMD ["flask","--app","main.py","run","--host=0.0.0.0"]
