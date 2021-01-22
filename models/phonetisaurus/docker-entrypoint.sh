#!/bin/sh

gunicorn --chdir /phonetisaurus/models/phonetisaurus wsgi:app -w 2 -b 0.0.0.0:5001
