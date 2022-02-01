#! /bin/bash
APP_PATH=`pwd`
cd models/research
export PYTHONPATH=${PWD}
cd APP_PATH
python final_app.py
