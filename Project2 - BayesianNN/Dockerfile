FROM --platform=linux/amd64 python:3.8-slim
RUN apt-get update
RUN apt-get -y install gcc
RUN apt-get install libssl-dev
ADD ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r /requirements.txt
WORKDIR /code
ADD * /code/
ADD pytransform /code/pytransform
WORKDIR /code
CMD python -u checker_client.py
