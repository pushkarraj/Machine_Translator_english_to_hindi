import yaml
import pandas as pd
import argparse


def read_params(config_path):
    with open(config_path) as f:
        params = yaml.safe_load(f)
        return params


def get_data(config_path):
    config = read_params(config_path)
    data_path = config['data_source']['master_file']
    df = pd.read_csv(data_path, sep=',', encoding='utf8',index_col=0)
    return df

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="config/params.yaml", )
    args.add_argument("--data_path", type=str, default="./data")
    parsed_args = args.parse_args()
    print(parsed_args.config)
    data = get_data(parsed_args.config)
    print(data)

