import numpy
import tensorflow
import time

# local
from data.last_data import *
from compute_etl_func import *
from configuration.man_v3_woman_v4_size_configuration import *
from util_log import *

# prod
# from data.last_data import *
# from compute_etl_func import *
# from configuration.man_v3_woman_v4_size_configuration import *
# from util_log import *

# 日志获取
logger = get_logger(LOG_FILE_PATH,"size-predict-log")

# load model
# man
StandardScaler_mlp_man_0 = pandas.read_pickle(StandardScaler_mlp_man_0)
StandardScaler_mlp_man_1 = pandas.read_pickle(StandardScaler_mlp_man_1)
StandardScaler_mlp_man_2 = pandas.read_pickle(StandardScaler_mlp_man_2)

man_lgbm_0 = pandas.read_pickle(lgbm_man_0)
man_lgbm_1 = pandas.read_pickle(lgbm_man_1)
man_lgbm_2 = pandas.read_pickle(lgbm_man_2)

StandardScaler_lgbm_man_0 = pandas.read_pickle(StandardScaler_lgbm_man_0)
StandardScaler_lgbm_man_1 = pandas.read_pickle(StandardScaler_lgbm_man_1)
StandardScaler_lgbm_man_2 = pandas.read_pickle(StandardScaler_lgbm_man_2)

# woman
StandardScaler_mlp_woman_0 = pandas.read_pickle(StandardScaler_mlp_woman_0)
StandardScaler_mlp_woman_1 = pandas.read_pickle(StandardScaler_mlp_woman_1)
StandardScaler_mlp_woman_2 = pandas.read_pickle(StandardScaler_mlp_woman_2)
StandardScaler_mlp_woman_3 = pandas.read_pickle(StandardScaler_mlp_woman_3)
StandardScaler_mlp_woman_4 = pandas.read_pickle(StandardScaler_mlp_woman_4)

woman_lgbm_0 = pandas.read_pickle(lgbm_woman_0)
woman_lgbm_1 = pandas.read_pickle(lgbm_woman_1)
woman_lgbm_2 = pandas.read_pickle(lgbm_woman_2)
woman_lgbm_3 = pandas.read_pickle(lgbm_woman_3)
woman_lgbm_4 = pandas.read_pickle(lgbm_woman_4)

StandardScaler_lgbm_woman_0 = pandas.read_pickle(StandardScaler_lgbm_woman_0)
StandardScaler_lgbm_woman_1 = pandas.read_pickle(StandardScaler_lgbm_woman_1)
StandardScaler_lgbm_woman_2 = pandas.read_pickle(StandardScaler_lgbm_woman_2)
StandardScaler_lgbm_woman_3 = pandas.read_pickle(StandardScaler_lgbm_woman_3)
StandardScaler_lgbm_woman_4 = pandas.read_pickle(StandardScaler_lgbm_woman_4)

# 模型预测
def size_predict_man(data):
    k = []

    temp_data_0 = StandardScaler_mlp_man_0.transform(data).astype(numpy.float32)
    tensorflow.reset_default_graph()
    with tensorflow.Session() as sess:
        Saver = tensorflow.train.import_meta_graph(man_meta_0)
        Saver.restore(sess,man_0)
        x = tensorflow.get_default_graph().get_tensor_by_name("x:0")
        y = tensorflow.get_default_graph().get_tensor_by_name("y:0")
        result = sess.run(tensorflow.nn.softmax(y), feed_dict={x: temp_data_0})
        k.append(result[:, 1].reshape(-1, 1)[0])

    temp_data_1 = StandardScaler_mlp_man_1.transform(data).astype(numpy.float32)
    tensorflow.reset_default_graph()
    with tensorflow.Session() as sess:
        Saver = tensorflow.train.import_meta_graph(man_meta_1)
        Saver.restore(sess, man_1)
        x = tensorflow.get_default_graph().get_tensor_by_name("x:0")
        y = tensorflow.get_default_graph().get_tensor_by_name("y:0")
        result = sess.run(tensorflow.nn.softmax(y), feed_dict={x: temp_data_1})
        k.append(result[:, 1].reshape(-1, 1)[0])

    temp_data_2 = StandardScaler_mlp_man_2.transform(data).astype(numpy.float32)
    tensorflow.reset_default_graph()
    with tensorflow.Session() as sess:
        Saver = tensorflow.train.import_meta_graph(man_meta_2)
        Saver.restore(sess, man_2)
        x = tensorflow.get_default_graph().get_tensor_by_name("x:0")
        y = tensorflow.get_default_graph().get_tensor_by_name("y:0")
        result = sess.run(tensorflow.nn.softmax(y), feed_dict={x: temp_data_2})
        k.append(result[:, 1].reshape(-1, 1)[0])


    temp_0 = StandardScaler_lgbm_man_0.transform(data)
    y = man_lgbm_0.predict_proba(temp_0)
    k.append(y[:, 1].reshape(-1, 1)[0])


    temp_1 = StandardScaler_lgbm_man_1.transform(data)
    y = man_lgbm_1.predict_proba(temp_1)
    k.append(y[:, 1].reshape(-1, 1)[0])


    temp_2 = StandardScaler_lgbm_man_2.transform(data)
    y = man_lgbm_2.predict_proba(temp_2)
    k.append(y[:, 1].reshape(-1, 1)[0])



    temp_data_3=numpy.array(k).reshape(-1,2*3).astype(numpy.float32)

    tensorflow.reset_default_graph()
    with tensorflow.Session() as sess:
        Saver = tensorflow.train.import_meta_graph(man_meta_stack)
        Saver.restore(sess, man_stack)
        x = tensorflow.get_default_graph().get_tensor_by_name("x:0")
        y = tensorflow.get_default_graph().get_tensor_by_name("y:0")
        result = sess.run(tensorflow.nn.softmax(y), feed_dict={x: temp_data_3})
        # result=json.dumps({'size':float(result[0][1])})
        result = float(result[0][1])
    return result

# 模型预测
def size_predict_woman(data):
    k = []
    temp_data_0 = StandardScaler_mlp_woman_0.transform(data).astype(numpy.float32)
    tensorflow.reset_default_graph()
    with tensorflow.Session() as sess:
        Saver = tensorflow.train.import_meta_graph(woman_meta_0)
        Saver.restore(sess, woman_0)
        x = tensorflow.get_default_graph().get_tensor_by_name("x:0")
        y = tensorflow.get_default_graph().get_tensor_by_name("y:0")
        result = sess.run(tensorflow.nn.softmax(y), feed_dict={x: temp_data_0})
        k.append(result[:, 1].reshape(-1, 1)[0])

    temp_data_1 = StandardScaler_mlp_woman_1.transform(data).astype(numpy.float32)
    tensorflow.reset_default_graph()
    with tensorflow.Session() as sess:
        Saver = tensorflow.train.import_meta_graph(woman_meta_1)
        Saver.restore(sess, woman_1)
        x = tensorflow.get_default_graph().get_tensor_by_name("x:0")
        y = tensorflow.get_default_graph().get_tensor_by_name("y:0")
        result = sess.run(tensorflow.nn.softmax(y), feed_dict={x: temp_data_1})
        k.append(result[:, 1].reshape(-1, 1)[0])

    temp_data_2 = StandardScaler_mlp_woman_2.transform(data).astype(numpy.float32)
    tensorflow.reset_default_graph()
    with tensorflow.Session() as sess:
        Saver = tensorflow.train.import_meta_graph(woman_meta_2)
        Saver.restore(sess, woman_2)
        x = tensorflow.get_default_graph().get_tensor_by_name("x:0")
        y = tensorflow.get_default_graph().get_tensor_by_name("y:0")
        result = sess.run(tensorflow.nn.softmax(y), feed_dict={x: temp_data_2})
        k.append(result[:, 1].reshape(-1, 1)[0])

    temp_data_3 = StandardScaler_mlp_woman_3.transform(data).astype(numpy.float32)
    tensorflow.reset_default_graph()
    with tensorflow.Session() as sess:
        Saver = tensorflow.train.import_meta_graph(woman_meta_3)
        Saver.restore(sess, woman_3)
        x = tensorflow.get_default_graph().get_tensor_by_name("x:0")
        y = tensorflow.get_default_graph().get_tensor_by_name("y:0")
        result = sess.run(tensorflow.nn.softmax(y), feed_dict={x: temp_data_3})
        k.append(result[:, 1].reshape(-1, 1)[0])

    temp_data_4 = StandardScaler_mlp_woman_4.transform(data).astype(numpy.float32)
    tensorflow.reset_default_graph()
    with tensorflow.Session() as sess:
        Saver = tensorflow.train.import_meta_graph(woman_meta_4)
        Saver.restore(sess, woman_4)
        x = tensorflow.get_default_graph().get_tensor_by_name("x:0")
        y = tensorflow.get_default_graph().get_tensor_by_name("y:0")
        result = sess.run(tensorflow.nn.softmax(y), feed_dict={x: temp_data_4})
        k.append(result[:, 1].reshape(-1, 1)[0])

    temp_0 = StandardScaler_lgbm_woman_0.transform(data)
    y = woman_lgbm_0.predict_proba(temp_0)
    k.append(y[:, 1].reshape(-1, 1)[0])

    temp_1 = StandardScaler_lgbm_woman_1.transform(data)
    y = woman_lgbm_1.predict_proba(temp_1)
    k.append(y[:, 1].reshape(-1, 1)[0])

    temp_2 = StandardScaler_lgbm_woman_2.transform(data)
    y = woman_lgbm_2.predict_proba(temp_2)
    k.append(y[:, 1].reshape(-1, 1)[0])

    temp_3 = StandardScaler_lgbm_woman_3.transform(data)
    y = woman_lgbm_3.predict_proba(temp_3)
    k.append(y[:, 1].reshape(-1, 1)[0])

    temp_4 = StandardScaler_lgbm_woman_4.transform(data)
    y = woman_lgbm_4.predict_proba(temp_4)
    k.append(y[:, 1].reshape(-1, 1)[0])

    temp_data_5 = numpy.array(k).reshape(-1, 2 * 5).astype(numpy.float32)

    tensorflow.reset_default_graph()
    with tensorflow.Session() as sess:
        Saver = tensorflow.train.import_meta_graph(woman_meta_stack)
        Saver.restore(sess, woman_stack)
        x = tensorflow.get_default_graph().get_tensor_by_name("x:0")
        y = tensorflow.get_default_graph().get_tensor_by_name("y:0")
        result = sess.run(tensorflow.nn.softmax(y), feed_dict={x: temp_data_5})
        # result=json.dumps({'size':float(result[0][1])})
        result = float(result[0][1])
    return result

# 模型预测数据准备 脚数据6个维度
def man_v3_woman_v4_size_predict(data):
    (sex,foot) = foot_parase(data)
    logger.info('foot parase time:  ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    result_list = list()

    # size 模型计算
    if sex == 1:
        foot_last_list = foot_connect_last(foot,man_last)
        foot_last_list = get_etl_data_left_right_together(foot_last_list)
        for foot_last in foot_last_list:
            size = foot_last[0]
            result = size_predict_man([foot_last[1]])
            result_list.append((size, result))
    else:
        foot_last_list = foot_connect_last(foot,woman_last)
        foot_last_list = get_etl_data_left_right_together(foot_last_list)
        for foot_last in foot_last_list:
            size = foot_last[0]
            result = size_predict_woman([foot_last[1]])
            result_list.append((size, result))
    logger.info('foot connect last and size compute time:  ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.info(str(result_list))
    max_size = sorted(result_list,key=lambda size_score:size_score[1],reverse=True)[0][0]
    logger.info('get max size compute time:  ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    return max_size

