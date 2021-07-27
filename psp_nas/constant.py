import math

SEED = 123

PRUNE_FEATURE_ORDER = 1
PRUNE_NUM_NORM = 5
PRUNE_NUM_FEA_ENGINE = 10

PRUNE_NUM_FIRST = 8
PRUNE_NUM_SECOND = 8
PRUNE_NUM_LARGER = 12

TOP_K = 10
EACH_TOPK_RETRAIN_ROUND = 20
BEST_ARCH_RETRAIN_ROUND = 100

########## Controller Params ##########
CONTROLLER_LEAVES = 31
CONTROLLER_LR = 0.05

CONTROLLER_N = 200
CONTROLLER_M = 5000
CONTROLLER_M_LARGER = 8000

CONTROLLER_K = 60
PER_ROUND_ADD = 60

CONTROLLER_N_LARGER = 600
CONTROLLER_K_LARGER = 60


CONTROLLER_ITERATIONS = 5
CONTROLLER_NUM_BOOST_ROUND = 100

########## Child Model Params ###########
IN_DROP = 0.6
MULTI_LABEL = False
LR = 0.005
WEIGHT_DECAY = 5e-4

EPOCHS = 500
EARLY_STOP_SIZE = 30
BEST_ARC_RETRAIN_EPOCH = 500
# MAX_NOT_EARLY_STOP_EPOCH = 100
MAX_NOT_EARLY_STOP_EPOCH = BEST_ARC_RETRAIN_EPOCH // 2

LAYERS_OF_CHILD_MODEL = 2

########## GBDT Params #############
GBDT_PARAMS = \
    {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2'},
        'min_data_in_leaf': 7,
        'num_leaves': CONTROLLER_LEAVES,
        'learning_rate': CONTROLLER_LR,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'verbose': 0
    }


########## Default Params ############
CUDA = True
