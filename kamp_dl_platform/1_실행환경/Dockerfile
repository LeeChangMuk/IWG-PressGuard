# Base docker image 정의 (scikit-learn 1.8 / pandas 3.0 은 Python 3.11 이상 필요)
FROM python:3.12-slim

# 사용자 임의 빌드 구문 추가: 결과 그림의 한글 글꼴
RUN apt-get update
RUN apt-get install -y --no-install-recommends fonts-nanum

#플랫폼 사용 필요
ADD ./requirements.txt /work/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /work/requirements.txt
