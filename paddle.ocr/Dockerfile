FROM ghcr.io/giim-hf-lab/hailab/base:ubuntu20.04-cuda11.2-cudnn8-cn

ARG PORT=80
ARG PADDLE_INDEX="https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html"

RUN apt-get update
RUN apt-get install -y ffmpeg

RUN /opt/conda/bin/python3 -m pip install --no-color --no-input -U msgpack hypercorn fastapi numpy protobuf==3.20.0 opencv-python
RUN /opt/conda/bin/python3 -m pip install --no-color --no-input -U -f $PADDLE_INDEX paddlepaddle-gpu==2.3.0.post112 paddleocr

COPY app/ /root/app/

RUN rm -rf /var/lib/apt/lists

EXPOSE $PORT

WORKDIR /root

CMD /opt/conda/bin/python3 -m app -L . -p $PORT
