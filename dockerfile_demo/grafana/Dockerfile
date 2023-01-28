# getting base image from jenkins
FROM grafana/grafana

# This is optional – give your name & email ID
MAINTAINER Vishali vishalisrinivasan97@gmail.com

# Replace company and port no accordingly
ENV HTTP_PROXY "http://company.com:port_no"
ENV HTTPS_PROXY "http://company.com:port_no"

USER root

RUN printf 'Acquire::http::Proxy "http://company.com:port_no";\n Acquire::https::Proxy "http://company.com:port_no";\n' >> /etc/grafana/grafana.ini

EXPOSE 5000:3000
