鞋码预测接口
   接口地址：http://54.223.27.88:5000/size_predict

   post请求
   数据格式：
    {
        "result": {
        "foot_length_original": {
            "left": 253.234,
            "right": 252.811
        },
        "pocket_heel_girth": {
            "left": 317.803,
            "right": 315.047
        },
        "back_girth": {
            "left": 247.748,
            "right": 244.825
        },
        "plantar_girth": {
            "left": 233.666,
            "right": 232.289
        },
        "first_metatarsophalangeal_joint_length": {
            "left": 185,
            "right": 185
        },
        "metatarsophalangeal_girth": {
            "left": 172.941,
            "right": 168.136
        },
        "sex": 2
    }}
    输出 概率最大的码段：
    245

部分需要下载的包：
pip3 install flask
pip3 install request
pip3 install tensorflow