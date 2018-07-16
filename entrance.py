#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@author: zhanghao
@contact: zhanghao@epoque.shoes
@file: entrance.py
@time: 2018/5/2/002  18:03
"""

import time

from flask import Flask, request

# local
from compute.man_v3_woman_v4_size import *
from util_log import *
from configuration.man_v3_woman_v4_size_configuration import *

# prod
# from compute.man_v3_woman_v4_size import *
# from util_log import *
# from configuration.man_v3_woman_v4_size_configuration import *



# 日志获取
logger = get_logger(LOG_FILE_PATH,"size-predict-log")

app = Flask(__name__)

# 鞋码预测接口
@app.route('/size_predict', methods=['POST'])
def man_woman_size_compute():
    try:
        logger.info('start request time:  ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # 需要从request对象读取表单内容
        foot = request.form["result"]
        # logger.info(str(foot))
        size_predict_result = None
        if foot != None:
            size_predict_result = man_v3_woman_v4_size_predict(foot)
        logger.info(str(size_predict_result))
        return str(size_predict_result)
    except Exception as e:
        logger.error(str(e))
if __name__ == '__main__':
    app.run()
    # app.run(host='0.0.0.0', port=5000)