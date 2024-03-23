FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN apt-get install -y && pip install --upgrade pip && pip install -r requirements.txt
EXPOSE 80
CMD python app.py