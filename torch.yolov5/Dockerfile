FROM nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04

ADD https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh /tmp/miniconda.sh
RUN ["/bin/bash", "/root/miniconda.sh", "-b", "-p", "/opt/conda"]
RUN ["rm", "/tmp/miniconda.sh"]

RUN ["/opt/conda/bin/conda", "install", "--yes", "--update-all", "python=3.9", "pip"]
RUN ["/opt/conda/bin/conda", "install", "--yes", "-c", "conda-forge", "fastapi"]
RUN ["/opt/conda/bin/conda", "install", "--yes", "-c", "pytorch", "pytorch", "torchvision"]
RUN ["/opt/conda/bin/conda", "clean", "--yes", "--all"]

WORKDIR /root
