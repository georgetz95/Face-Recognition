FROM shosoar/alpine-python-opencv
MAINTAINER Giorgos Tzimas


WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE  80

CMD [ "python", "./app.py" ]