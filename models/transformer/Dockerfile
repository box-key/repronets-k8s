FROM python:3.7-slim-buster

ENV VERSION = 1
ENV PATCHLEVEL = 0
ENV SUBLEVEL = 0

WORKDIR /transformer

# copy trained models
COPY model_store/arabic/ctranslate2_released /transformer/model_store/arabic/ctranslate2_released
COPY model_store/chinese/ctranslate2_released /transformer/model_store/chinese/ctranslate2_released
COPY model_store/hebrew/ctranslate2_released /transformer/model_store/hebrew/ctranslate2_released
COPY model_store/katakana/ctranslate2_released /transformer/model_store/katakana/ctranslate2_released
COPY model_store/korean/ctranslate2_released /transformer/model_store/korean/ctranslate2_released
COPY model_store/russian/ctranslate2_released /transformer/model_store/russian/ctranslate2_released

# copy other files
COPY models/transformer/requirements.txt /transformer/requirements.txt
COPY models/transformer/predictor.py /transformer/models/transformer/predictor.py
COPY models/transformer/wsgi.py /transformer/models/transformer/wsgi.py
COPY models/transformer/docker-entrypoint.sh  /usr/local/bin/docker-entrypoint.sh

# install python dependencies
RUN pip3 install -r /transformer/requirements.txt

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT [ "docker-entrypoint.sh" ]
