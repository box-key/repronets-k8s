FROM python:3.7-slim-buster

ENV VERSION = 1
ENV PATCHLEVEL = 0
ENV SUBLEVEL = 0

WORKDIR /phonetisaurus

# copy trained models
COPY model_store/arabic_ps_1 /phonetisaurus/model_store/arabic_ps_1
COPY model_store/chinese_ps_1 /phonetisaurus/model_store/chinese_ps_1
COPY model_store/hebrew_ps_1  /phonetisaurus/model_store/hebrew_ps_1
COPY model_store/katakana_ps_1 /phonetisaurus/model_store/katakana_ps_1
COPY model_store/korean_ps_1 /phonetisaurus/model_store/korean_ps_1
COPY model_store/russian_ps_1 /phonetisaurus/model_store/russian_ps_1

# copy other files
COPY models/phonetisaurus/requirements.txt /phonetisaurus/requirements.txt
COPY models/phonetisaurus/predictor.py /phonetisaurus/models/phonetisaurus/predictor.py
COPY models/phonetisaurus/wsgi.py /phonetisaurus/models/phonetisaurus/wsgi.py
COPY models/phonetisaurus/docker-entrypoint.sh  /usr/local/bin/docker-entrypoint.sh

# install phonetisaurus
# source: https://github.com/AdolfVonKleist/Phonetisaurus
RUN apt-get update && apt-get install -y \
    git \
    g++ \
    autoconf-archive \
    make \
    libtool \
    python3-setuptools \
    python3-dev \
    python3-pip \
    gfortran \
    wget \
    tar \
    gawk

RUN cd /phonetisaurus

# grab and install OpenFst-1.6.2
RUN wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.2.tar.gz && \
    tar -xvzf openfst-1.6.2.tar.gz && \
    cd openfst-1.6.2 && \
    ./configure --enable-static --enable-shared --enable-far --enable-ngram-fsts && \
    make -j 4 && \
    make install && \
    ldconfig && \
    cd

# compile with python3 bindings
RUN git clone https://github.com/AdolfVonKleist/Phonetisaurus.git && \
    cd Phonetisaurus && \
    pip3 install pybindgen && \
    PYTHON=python3 ./configure --enable-python && \
    make && \
    make install && \
    cd python && \
    cp ../.libs/Phonetisaurus.so . && \
    python3 setup.py install && \
    cd

# grab and install mitlm
RUN git clone https://github.com/mitlm/mitlm.git && \
    cd mitlm && \
    ./autogen.sh && \
    make && \
    make install && \
    cd

# install python dependencies
RUN pip3 install -r /phonetisaurus/requirements.txt
# copy compiled code into model directory
#RUN cp /phonetisaurus/Phonetisaurus/.libs/Phonetisaurus.so /phonetisaurus/models/phonetisaurus/Phonetisaurus.so
RUN cp /phonetisaurus/Phonetisaurus/.libs/Phonetisaurus.so /phonetisaurus/Phonetisaurus.so
RUN cp /phonetisaurus/Phonetisaurus/python /phonetisaurus/python -r

# check installation
RUN phonetisaurus-apply --help

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT [ "docker-entrypoint.sh" ]

