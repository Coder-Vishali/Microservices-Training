FROM python:3.6

ENV HTTP_PROXY "http://company.com:port"
ENV HTTPS_PROXY "http://company.com:port"

EXPOSE 5000

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY app.py /app
CMD python app.py
