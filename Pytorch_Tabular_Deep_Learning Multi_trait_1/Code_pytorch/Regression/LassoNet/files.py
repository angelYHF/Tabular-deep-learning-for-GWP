from time import time
import os

def make_sure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def make_filename(param_dict, postfix):
    filename = f'{int(time())}'
    for k,v in param_dict.items():
        filename+=f'_{k}_{v}'
    
    return filename+postfix