FROM apache/airflow:2.7.0
COPY requirements.txt /requirements.txt


RUN pip install --user --upgrade pip
RUN pip install --no-cache-dir --user -r /requirements.txt

USER root

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1