import os
from lassonet.lassonet import LassoNetRegressorCV, plot_cv
import torch
from functools import partial
import matplotlib.pyplot as plt
from files import make_filename, make_sure_dir


def collate_params(bo_config, **params):
    """
    Hyper-parameters given by the Bayesian Optimizer should be collated according to parameter types(e.g. discrete/continous)
    Return the collated params dict.
    """
    bo_param_scope = bo_config['param_scope']
    collated_params = {}

    for k, v in params.items():
        if bo_param_scope[k]['is_discrete']:
            v = round(v)
        collated_params[k] = v

    return collated_params


def validate_params(**params):
    if params['hidden_layer2_features'] <= 0:
        params.pop('hidden_layer2_features')
    return params


def cv_eval_model(ds_tuple, bo_config, **params):
    train_feature, train_label, test_feature, test_label = ds_tuple

    params = collate_params(bo_config, **params)
    params = validate_params(**params)

    model = LassoNetRegressorCV(
        hidden_dims=(params['hidden_layer1_features'],
                     params['hidden_layer2_features'])
        if params['hidden_layer2_features'] > 0
        else (params['hidden_layer1_features'],),
        # lambda_start=0.1,
        # path_multiplier=1.2,
        M=params['M'],
        optim=(
            partial(torch.optim.Adam, lr=params['learning_rate']),
            partial(torch.optim.SGD, lr=params['learning_rate'], momentum=0.9),
        ),
        batch_size=params['batch_size'],
        device='cuda',
        random_state=42,
        torch_seed=42,
    )

    model.path(train_feature, train_label)
    testing_mse = model.score(test_feature, test_label)
    print("Best model mse", testing_mse)
    print("Lambda =", model.best_lambda_)

    plot_cv(model, test_feature, test_label)
    plt.savefig(os.path.join(make_sure_dir(
        bo_config['output_path']), make_filename(params, postfix='.png')))

    return -testing_mse