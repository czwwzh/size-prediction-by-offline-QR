import json


from configuration.man_v3_woman_v4_size_configuration_prod import *
from util_log import *


# 日志获取
logger = get_logger(LOG_FILE_PATH,"size-predict-log-2")

# 脚楦数据获取与处理函数定义
# 脚数据解析
def foot_parase(data):
    foot_result = dict()
    try:
        data = json.loads(data)
        for foot_attr in data:
            if foot_attr != "sex":
                foot_result[foot_attr+"_left"] =  data[foot_attr]["left"]
                foot_result[foot_attr + "_right"] = data[foot_attr]["right"]
        return (data["sex"],foot_result)
    except Exception as e:
        logger.info(str(e))

# 单个脚与多个楦连接
def foot_connect_last(foot,last_list):
    foot_last_list = list()
    for last in last_list:
        foot_last_list.append(dict(foot, **last))
    return foot_last_list

# 按指定顺序获取舒适度模型计算所需要的数据和用户信息
# get left  right data together for size data_compute
# return [[size,leftright]]
def get_etl_data_left_right_together(data):
    left_right_datas = list()
    for left_right_data in data:
        left_right = list()
        size = left_right_data['basicsize']
        for field in FOOT_LAST_ORDER_DIMENSIONS_DIM6:
            left_right.append(left_right_data[field])
        left_right_datas.append([size,left_right])
    return left_right_datas
