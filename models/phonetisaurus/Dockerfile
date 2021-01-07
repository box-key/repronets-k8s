FROM python:3.7-slim-buste

WORKDIR /phonetisaurus

# trained models
COPY model_store/arabic_ps_1 /
COPY model_store/chinese_ps_1 /
COPY model_store/hebrew_ps_1  /
COPY model_store/katakana_ps_1 /
COPY model_store/korean_ps_1 /
COPY model_store/russian_ps_1 /

# install phonetisaurus
# source: https://github.com/AdolfVonKleist/Phonetisaurus
RUN apt-get update && \
  apt-get install -y \
    git \
    g++ \
    autoconf-archive \
    make \
    libtool \
    python3-setuptools \
    python3-dev \
    python3-pip \
    gfortran

# grab and install OpenFst-1.6.2
RUN wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.2.tar.gz && \
  tar -xvzf openfst-1.6.2.tar.gz && \
  cd openfst-1.6.2 && \
  ./configure --enable-static --enable-shared --enable-far --enable-ngram-fsts && \
  make -j 4 && \
  make install && \
  cd

RUN pip3 install pybindgen
ENV PYTHON=python3 ./configure --enable-python

# compile with python3 bindings
RUN git clone https://github.com/AdolfVonKleist/Phonetisaurus.git && \
  cd Phonetisaurus && \
  pip3 install pybindgen && \
  PYTHON=python3 ./configure --enable-python && \
  make && \
  make install && \
  cd python && \
  cp ../.libs/Phonetisaurus.so . && \
  python3 setup.py instal && \
  cd

# grab and install mitlm
RUN git clone https://github.com/mitlm/mitlm.git && \
  cd mitlm && \
  ./autogen.sh && \
  make && \
  make install && \
  cd

# install python dependencies
COPY models/phonetisaurus/requirements.txt /phonetisaurus/requirements.txt
RUN pip3 install -r phonetisaurus/requirements.txt

# set port for this container
ENV PORT 1133
EXPOSE $PORT

ENTRYPOINT [ "gunicorn -w 4 predictor:app -b 127.0.0.1:$PORT" ]
