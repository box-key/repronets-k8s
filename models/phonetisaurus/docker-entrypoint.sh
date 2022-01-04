#!/bin/sh

gunicorn --chdir /phonetisaurus/models/phonetisaurus wsgi:app \
	--workers 1 \
	--bind 0.0.0.0:5000 \
	--timeout 300