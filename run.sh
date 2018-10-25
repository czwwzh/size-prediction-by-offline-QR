#!/bin/bash
source /home/ec2-user/virtualenv36/bin/activate
nohup python -u /root/projects/size-prediction-by-offline-QR/entrance.py >/dev/null 2>&1 &
