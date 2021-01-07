#!/bin/sh


gunicorn --chdir /phonetisaurus predictor:app -w 4 -b 0.0.0.0:5001
