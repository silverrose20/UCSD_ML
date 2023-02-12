FROM python:3.9

EXPOSE 8080

WORKDIR /app

COPY . /app

RUN pip3 install --upgrade pip
RUN apt-get update && apt-get -y install cmake
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt --no-cache-dir

#ENTRYPOINT ["streamlit", "run", "webrtc.py", "--server.port=8501", "--server.address=0.0.0.0"]
CMD streamlit run webrtc.py --server.port 8080 --browser.serverAddress="0.0.0.0"