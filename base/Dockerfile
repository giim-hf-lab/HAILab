FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

COPY sources.list /etc/apt/sources.list
COPY ubuntu.list /etc/apt/sources.list.d/ubuntu.list
RUN apt-get update
RUN apt-get install -y gcc

ADD https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh /tmp/miniconda.sh
COPY .condarc /root/.condarc
COPY pip.conf /root/.config/pip/pip.conf
RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda
RUN /opt/conda/bin/conda install --yes --update-all python=3.9 pip

RUN rm -f /tmp/miniconda.sh
RUN rm -rf /var/lib/apt/lists
