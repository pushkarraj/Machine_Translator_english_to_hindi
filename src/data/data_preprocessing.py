import yaml
import pandas as pd
import argparse
import re
import copy
import numpy as np
from tqdm import tqdm
from data_ingestion import read_params,get_data
from sklearn.model_selection import train_test_split


def english_remover(hindi_text):
    ids_to_remove = {}
    for _id, _t in tqdm(enumerate(hindi_text)):
        if len(re.findall(r'[a-zA-Z]', str(_t))) > 0:
            ids_to_remove[_id] = _t
        else:
            pass
    ids_to_keep = [i for i in range(len(hindi_text)) if i not in ids_to_remove.keys()]
    return ids_to_keep

def remove_sc(_line, lang="en"):
    if lang == "hi":
        _line = re.sub(r'[+\-*/@%>=;~{}×–`’"()_]', "", str(_line))
        _line = re.sub(r"(?:(\[)|(\])|(‘‘)|(’’))", '', str(_line))
    elif lang == "en":
        _line = re.sub(r'[+\-*/@%>=;~{}×–`’"()_|:]', "", str(_line))
        _line = re.sub(r"(?:(\[)|(\])|(‘‘)|(’’))", '', str(_line))
    return _line

def contractions(config_path):
    config= read_params(config_path)
    path=config['data_contractions']['contract_file']
    with open(path, "r") as inp_cont:
        contractions_list = inp_cont.read()
    contractions_list = [re.sub('["]', '', x).split(":") for x in re.sub(r"\s+", " ", re.sub(r"(.*{)|(}.*)", '', contractions_list)).split(',')]
    contractions_dict = dict((k.lower().strip(), re.sub('/.*', '', v).lower().strip()) for k, v in contractions_list)
    return contractions_dict

def clean_text(_text,lang,dic):
    if lang == "en":
        _text = remove_sc(_line=_text, lang=lang)
        contractions_dict=dic
        for cn in contractions_dict:
            _text = re.sub(cn, contractions_dict[cn], _text)
    elif lang == "hi":
        _text = remove_sc(_line=_text, lang=lang)

    return _text


def preprocess(config_path):
    config = read_params(config_path)
    df=get_data(config_path)
    id=english_remover(df["1"].to_list())
    df=df.loc[id]
    dic=contractions(config_path)

    tqdm.pandas()
    df["0"]=df["0"].progress_map(lambda x:clean_text(x,lang='en',dic=dic))
    df["1"]=df["1"].progress_map(lambda x:clean_text(x,lang='hi',dic=dic))

    df["eng_len"] = df["0"].str.len()
    df["hindi_len"] = df["1"].str.len()
    small_len_data = df.query('eng_len < 50 & hindi_len < 50')

    train_path=config["data_source"]["training_file"]
    val_path=config["data_source"]["validation_file"]
    train_set, val_set = train_test_split(small_len_data.loc[:, ["0", "1"]], test_size=0.1)
    train_set.to_csv(train_path, index=False)
    val_set.to_csv(val_path, index=False)

    # Small set
    train_path_sm = config["data_source"]["small_training_file"]
    val_path_sm = config["data_source"]["small_validation_file"]
    small_data = small_len_data.loc[:, ["0", "1"]].sample(n=150000)
    train_set_sm, val_set_sm = train_test_split(small_data, test_size=0.3)
    train_set_sm.to_csv(train_path_sm, index=False)
    val_set_sm.to_csv(val_path_sm, index=False)

    return df,train_set,val_set,train_set_sm,val_set_sm

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="config/params.yaml", )
    args.add_argument("--data_path", type=str, default="./data")
    parsed_args = args.parse_args()
    print(parsed_args)
    preprocess(parsed_args.config)