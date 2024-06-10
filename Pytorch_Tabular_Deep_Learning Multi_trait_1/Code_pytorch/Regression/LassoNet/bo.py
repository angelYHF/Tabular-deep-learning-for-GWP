import bayes_opt
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from omegaconf import OmegaConf
from model import cv_eval_model
import os
from functools import partial
from files import make_filename, make_sure_dir

def parse_bo_param_scope(bo_param_scope):
    scope = {}
    for k,v in bo_param_scope.items():
        sorted_value = v['value']
        sorted_value.sort()
        value_range_tuple = (sorted_value[0],sorted_value[-1])
        scope[k]=value_range_tuple
    
    return scope

def bo_search(dataset_tuple, bo_config_filepath):
    """
    Return optimum MSE point
    """
    bo_config = OmegaConf.load(bo_config_filepath)
    bo_setting = bo_config['setting']

    optimizer = bayes_opt.BayesianOptimization(
        f=partial(cv_eval_model, ds_tuple=dataset_tuple, bo_config=bo_config),
        pbounds=parse_bo_param_scope(bo_config['param_scope']),
        random_state=bo_setting['random_state_seed']
    )
    bayesianOptCheckpointPath = os.path.join(make_sure_dir(bo_config['checkpoint_path']), make_filename(bo_setting, postfix=''))
    logger = JSONLogger(path=bayesianOptCheckpointPath)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=bo_setting['init_random_points'],
        n_iter=bo_setting['bo_iterations'],
    )

    # Max -MSE value:
    optim_mse=-optimizer.max['target']

    return optim_mse, optimizer.max