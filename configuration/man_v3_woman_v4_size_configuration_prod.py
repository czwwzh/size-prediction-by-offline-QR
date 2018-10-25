import pandas

# 1.模型计算所用变量定义 prod
# man
StandardScaler_mlp_man_0 = "/root/projects/man_v3_woman_v4_size/StandardScaler_dim6_mlp_man_0"
StandardScaler_mlp_man_1 = "/root/projects/man_v3_woman_v4_size/StandardScaler_dim6_mlp_man_1"
StandardScaler_mlp_man_2 = "/root/projects/man_v3_woman_v4_size/StandardScaler_dim6_mlp_man_2"

man_meta_0 = "/root/projects/man_v3_woman_v4_size/mlp_dim6_man_0.meta"
man_meta_1 = "/root/projects/man_v3_woman_v4_size/mlp_dim6_man_1.meta"
man_meta_2 = "/root/projects/man_v3_woman_v4_size/mlp_dim6_man_2.meta"
man_meta_stack = "/root/projects/man_v3_woman_v4_size/stack_dim6_man.meta"
man_0 = "/root/projects/man_v3_woman_v4_size/mlp_dim6_man_0"
man_1 = "/root/projects/man_v3_woman_v4_size/mlp_dim6_man_1"
man_2 = "/root/projects/man_v3_woman_v4_size/mlp_dim6_man_2"
man_stack = "/root/projects/man_v3_woman_v4_size/stack_dim6_man"

lgbm_man_0 = "/root/projects/man_v3_woman_v4_size/lgbm_dim6_man_0"
lgbm_man_1 = "/root/projects/man_v3_woman_v4_size/lgbm_dim6_man_1"
lgbm_man_2 = "/root/projects/man_v3_woman_v4_size/lgbm_dim6_man_2"

StandardScaler_lgbm_man_0 = "/root/projects/man_v3_woman_v4_size/StandardScaler_dim6_lgbm_man_0"
StandardScaler_lgbm_man_1 = "/root/projects/man_v3_woman_v4_size/StandardScaler_dim6_lgbm_man_1"
StandardScaler_lgbm_man_2 = "/root/projects/man_v3_woman_v4_size/StandardScaler_dim6_lgbm_man_2"


# woman
StandardScaler_mlp_woman_0 = "/root/projects/man_v3_woman_v4_size/StandardScaler_dim6_mlp_woman_0"
StandardScaler_mlp_woman_1 = "/root/projects/man_v3_woman_v4_size/StandardScaler_dim6_mlp_woman_1"
StandardScaler_mlp_woman_2 = "/root/projects/man_v3_woman_v4_size/StandardScaler_dim6_mlp_woman_2"
StandardScaler_mlp_woman_3 = "/root/projects/man_v3_woman_v4_size/StandardScaler_dim6_mlp_woman_3"
StandardScaler_mlp_woman_4 = "/root/projects/man_v3_woman_v4_size/StandardScaler_dim6_mlp_woman_4"

woman_meta_0 = "/root/projects/man_v3_woman_v4_size/mlp_dim6_woman_0.meta"
woman_meta_1 = "/root/projects/man_v3_woman_v4_size/mlp_dim6_woman_1.meta"
woman_meta_2 = "/root/projects/man_v3_woman_v4_size/mlp_dim6_woman_2.meta"
woman_meta_3 = "/root/projects/man_v3_woman_v4_size/mlp_dim6_woman_3.meta"
woman_meta_4 = "/root/projects/man_v3_woman_v4_size/mlp_dim6_woman_4.meta"
woman_meta_stack = "/root/projects/man_v3_woman_v4_size/stack_dim6_woman.meta"
woman_0 = "/root/projects/man_v3_woman_v4_size/mlp_dim6_woman_0"
woman_1 = "/root/projects/man_v3_woman_v4_size/mlp_dim6_woman_1"
woman_2 = "/root/projects/man_v3_woman_v4_size/mlp_dim6_woman_2"
woman_3 = "/root/projects/man_v3_woman_v4_size/mlp_dim6_woman_3"
woman_4 = "/root/projects/man_v3_woman_v4_size/mlp_dim6_woman_4"
woman_stack = "/root/projects/man_v3_woman_v4_size/stack_dim6_woman"

lgbm_woman_0 = "/root/projects/man_v3_woman_v4_size/lgbm_dim6_woman_0"
lgbm_woman_1 = "/root/projects/man_v3_woman_v4_size/lgbm_dim6_woman_1"
lgbm_woman_2 = "/root/projects/man_v3_woman_v4_size/lgbm_dim6_woman_2"
lgbm_woman_3 = "/root/projects/man_v3_woman_v4_size/lgbm_dim6_woman_3"
lgbm_woman_4 = "/root/projects/man_v3_woman_v4_size/lgbm_dim6_woman_4"

StandardScaler_lgbm_woman_0 = "/root/projects/man_v3_woman_v4_size/StandardScaler_dim6_lgbm_woman_0"
StandardScaler_lgbm_woman_1 = "/root/projects/man_v3_woman_v4_size/StandardScaler_dim6_lgbm_woman_1"
StandardScaler_lgbm_woman_2 = "/root/projects/man_v3_woman_v4_size/StandardScaler_dim6_lgbm_woman_2"
StandardScaler_lgbm_woman_3 = "/root/projects/man_v3_woman_v4_size/StandardScaler_dim6_lgbm_woman_3"
StandardScaler_lgbm_woman_4 = "/root/projects/man_v3_woman_v4_size/StandardScaler_dim6_lgbm_woman_4"


# 2.size data_compute attributes
SIZE_DIMENSION_NAME_DIM6 = '/root/projects/man_v3_woman_v4_size/size_dimension_name_dim6'

# 3.日志文件地址
LOG_FILE_PATH ='/home/ec2-user/zhanghao/log/size_predict_offine_'

# 4.size模型所需字段顺序定义
FOOT_LAST_ORDER_DIMENSIONS_DIM6 = pandas.read_pickle(SIZE_DIMENSION_NAME_DIM6)