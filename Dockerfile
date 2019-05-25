FROM ubuntu:latest

RUN apt-get update -y && apt-get install -y python-pip python-dev build-essential curl
RUN pip install flask nltk pandas numpy scikit-learn
RUN  mkdir -p /home/qwerty/ex/
COPY ex/ /home/qwerty/ex/ 
WORKDIR /home/qwerty/ex/
RUN chmod 644 /home/qwerty/ex/run.py
EXPOSE 5000

CMD ["python", "/home/qwerty/ex/run.py"]