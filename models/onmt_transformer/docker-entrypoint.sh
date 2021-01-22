#!/bin/sh

gunicorn --chdir /transformer/models/onmt_transformer wsgi:app -w 2 -b 0.0.0.0:5002
