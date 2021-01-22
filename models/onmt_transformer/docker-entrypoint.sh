#!/bin/sh

gunicorn --chdir /transformer predictor:get_app -w 4 -b 0.0.0.0:5002
