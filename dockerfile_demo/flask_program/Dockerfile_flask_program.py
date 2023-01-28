from flask import Flask

app = Flask(__name__)

# http://127.0.0.1:4002/welcome

@app.route("/welcome")

def welcome():
   return "Welcome To Vishali's Page"

# launch the develoment server

app.run(host='0.0.0.0',port=4005)

'''
Docker file for this file
FROM python:latest
ENV HTTP_PROXY "http://<company.com>:<port>"
ENV HTTPS_PROXY "http://<company.com>:<port>"
RUN mkdir -p /app
WORKDIR /app
COPY ./requirements.txt /app/
COPY ./main.py /app/
RUN pip install -r requirements.txt
EXPOSE 4005
CMD ["python","main.py"]
'''