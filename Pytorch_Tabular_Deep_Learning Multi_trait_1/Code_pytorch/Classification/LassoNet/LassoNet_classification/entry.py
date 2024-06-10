bo_config_file = 'param_space.yaml'
ds_file = 'wheat-hitopt.csv'

from data import load_wheat_csv_ds
from bo import bo_search

def main():
    ds_tuple = load_wheat_csv_ds(ds_file, testing_set_ratio=0.1)
    bo_search(ds_tuple, bo_config_file)

if __name__ == "__main__":
    main()