FROM python:3.10-slim
RUN apt-get update
RUN apt-get install -y --no-install-recommends fonts-nanum
#플랫폼 사용 필요
ADD ./requirements.txt /work/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /work/requirements.txt
