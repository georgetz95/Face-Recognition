FROM python:3.7-alpine

WORKDIR /Face-Recognition
ADD . /Face-Recognition
RUN pip install -r requirements.txt
CMD ["python", "app.py"]