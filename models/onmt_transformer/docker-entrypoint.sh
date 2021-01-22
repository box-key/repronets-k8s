#!/bin/sh

gunicorn --chdir /transformer/models/onmt_transformer wsgi:app -w 4 -b 0.0.0.0:5002
