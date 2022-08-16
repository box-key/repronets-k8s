# python 3.8.10 is required for torch==1.6.0
FROM python:3.8.10-slim-buster

ENV VERSION = 1
ENV PATCHLEVEL = 0
ENV SUBLEVEL = 1

WORKDIR /transformer

# set variables
ARG LANGUAGE unk
ENV LANGUAGE_NAME ${LANGUAGE}
RUN echo "Language = ${LANGUAGE}"

# copy trained models
COPY model_store/${LANGUAGE}/transformer/ctranslate2_released /transformer/model_store/${LANGUAGE}/transformer/ctranslate2_released

# copy other files
COPY models/transformer/requirements.txt /transformer/requirements.txt
COPY models/transformer/predictor.py /transformer/models/transformer/predictor.py
COPY models/transformer/wsgi.py /transformer/models/transformer/wsgi.py
COPY models/transformer/docker-entrypoint.sh  /usr/local/bin/docker-entrypoint.sh

# install python dependencies
RUN pip3 install -r /transformer/requirements.txt

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT [ "docker-entrypoint.sh" ]
