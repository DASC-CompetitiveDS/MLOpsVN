FROM python:3.10.6-slim

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -yq install libgomp1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# deployment/model_predictor/
ADD requirements.txt .
RUN pip install --upgrade wheel setuptools pip
RUN pip install -r requirements.txt

COPY ./src /sample_solution/src
WORKDIR /sample_solution