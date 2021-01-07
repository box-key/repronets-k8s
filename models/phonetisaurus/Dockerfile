FROM python:3.7-slim-buster

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
COPY models/phonetisaurus/predictor.py /phonetisaurus/predictor.py
COPY models/phonetisaurus/docker-entrypoint.sh  /usr/local/bin/docker-entrypoint.sh
COPY phonetisaurus-0.3.0-py3-none-manylinux1_x86_64.whl /phonetisaurus/phonetisaurus-0.3.0-py3-none-manylinux1_x86_64.whl

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
    tar

RUN cd /phonetisaurus

# grab and install OpenFst-1.6.2
RUN wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.2.tar.gz && \
    tar -xvzf openfst-1.6.2.tar.gz && \
    cd openfst-1.6.2 && \
    ./configure --enable-static --enable-shared --enable-far --enable-ngram-fsts && \
    make -j 4 && \
    make install && \
    cd

RUN pip3 install pybindgen

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
# install package from source (there is a package with the same name on pypi)
# link: https://pypi.org/project/phonetisaurus/#files
RUN pip3 install /phonetisaurus/phonetisaurus-0.3.0-py3-none-manylinux1_x86_64.whl --force-reinstall

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT [ "docker-entrypoint.sh" ]
