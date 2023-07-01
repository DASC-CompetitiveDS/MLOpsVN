FROM python:3.9

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -yq install libgomp1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./requirements.txt .
RUN pip install --upgrade wheel setuptools pip
RUN pip install -r requirements.txt

COPY ./src /sample_solution/src
WORKDIR /sample_solution
