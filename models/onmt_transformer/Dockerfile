FROM python:3.7-slim-buster

WORKDIR /transformer

# copy trained models
COPY model_store/arabic/ctranslate2_released /transformer/model_store/arabic/ctranslate2_released
COPY model_store/chinese/ctranslate2_released /transformer/model_store/chiense/ctranslate2_released
COPY model_store/hebrew/ctranslate2_released /transformer/model_store/hebrew/ctranslate2_released
COPY model_store/katakana/ctranslate2_released /transformer/model_store/katakana/ctranslate2_released
COPY model_store/korean/ctranslate2_released /transformer/model_store/korean/ctranslate2_released
COPY model_store/russian/ctranslate2_released /transformer/model_store/russian/ctranslate2_released

# copy other files
COPY models/onmt_transformer/requirements.txt /transformer/requirements.txt
COPY models/onmt_transformer/predictor.py /transformer/predictor.py
COPY models/onmt_transformer/docker-entrypoint.sh  /usr/local/bin/docker-entrypoint.sh

# install python dependencies
RUN pip3 install -r /transformer/requirements.txt

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT [ "docker-entrypoint.sh" ]
