import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import copy

# Packages for data generator & preparation
from torchtext.data import Field,TabularDataset,BucketIterator
import spacy
import sys
from indicnlp import common
from indicnlp.tokenize import indic_tokenize



# Packages for model building & inferences
from src.models.transformer import  Transformer
from src.training.trainer_utils import load_checkpoint

# Data Prep
# Settings for handling english text
spacy_eng = spacy.load("en_core_web_sm")

# Training & Evaluation
device = torch.device("cpu")

def beam_search(sentence, model, src_field, src_tokenizer, trg_field, trg_vcb_sz, k, max_ts=50, device=device):
    # Tokenize the input sentence
    sentence_tok = src_tokenizer(sentence)

    # Add <sos> and <eos> in beginning and end respectively
    sentence_tok.insert(0, src_field.init_token)
    sentence_tok.append(src_field.eos_token)

    # Converting text to indices
    src_tok = torch.tensor([src_field.vocab.stoi[token] for token in sentence_tok], dtype=torch.long).unsqueeze(0).to(device)
    trg_tok = torch.tensor([trg_field.vocab.stoi[trg_field.init_token]], dtype=torch.long).unsqueeze(0).to(device)

    # Setting 'eos' flag for target sentence
    eos = trg_field.vocab.stoi[trg_field.eos_token]

    # Store for top 'k' translations
    trans_store = {}

    store_seq_id = None
    store_seq_prob = None
    for ts in range(max_ts):
        if ts == 0:
            with torch.no_grad():
                out = model(src_tok, trg_tok)  # [1, trg_vcb_sz]
            topk = torch.topk(torch.log(torch.softmax(out, dim=-1)), dim=-1, k=k)
            seq_id = torch.empty(size=(k, ts + 2), dtype=torch.long)
            seq_id[:, :ts + 1] = trg_tok
            seq_id[:, ts + 1] = topk.indices
            seq_prob = topk.values.squeeze()
            # print(seq_prob.shape)
            if eos in seq_id[:, ts + 1]:
                trans_store[seq_prob[seq_id[:, ts + 1] == eos].squeeze()] = seq_id[seq_id[:, ts + 1] == eos, :].squeeze()
                store_seq_id = copy.deepcopy(seq_id[seq_id[:, ts + 1] != eos, :]).to(device)
                store_seq_prob = copy.deepcopy(seq_prob[seq_id[:, ts + 1] != eos].squeeze()).to(device)
            else:
                store_seq_id = copy.deepcopy(seq_id).to(device)
                store_seq_prob = copy.deepcopy(seq_prob).to(device)
        else:
            src_tok = src_tok.squeeze()
            src = src_tok.expand(size=(store_seq_id.shape[-2], len(src_tok))).to(device)
            with torch.no_grad():
                out = model(src, store_seq_id)
            out = torch.log(torch.softmax(out[:, -1, :], dim=-1))  # [k, trg_vcb_sz]
            all_comb = (store_seq_prob.view(-1, 1) + out).view(-1)
            all_comb_idx = torch.tensor([(x, y) for x in range(store_seq_id.shape[-2]) for y in range(trg_vcb_sz)])
            topk = torch.topk(all_comb, dim=-1, k=k)
            top_seq_id = all_comb_idx[topk.indices.squeeze()]
            top_seq_prob = topk.values
            seq_id = torch.empty(size=(k, ts + 2), dtype=torch.long)
            seq_id[:, :ts + 1] = torch.tensor([store_seq_id[i.tolist()].tolist() for i, y in top_seq_id])
            seq_id[:, ts + 1] = torch.tensor([y.tolist() for i, y in top_seq_id])
            seq_prob = top_seq_prob
            if eos in seq_id[:, ts + 1]:
                trans_store[seq_prob[seq_id[:, ts + 1] == eos].squeeze()] = seq_id[seq_id[:, ts + 1] == eos, :].squeeze()
                store_seq_id = copy.deepcopy(seq_id[seq_id[:, ts + 1] != eos, :]).to(device)
                store_seq_prob = copy.deepcopy(seq_prob[seq_id[:, ts + 1] != eos].squeeze()).to(device)
            else:
                store_seq_id = copy.deepcopy(seq_id).to(device)
                store_seq_prob = copy.deepcopy(seq_prob).to(device)
        if len(trans_store) == k:
            break

    if len(trans_store) == 0:
        best_translation = store_seq_id[0]
    else:
        best_translation = trans_store[max(trans_store)]
    return " ".join([trg_field.vocab.itos[w] for w in best_translation[1:]])


def get_translation(sentence):
    # Defining Tokenizer
    def tokenize_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def tokenize_hindi(text):
        return [tok for tok in indic_tokenize.trivial_tokenize(text)]

    # Defining Field
    english_txt = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")
    hindi_txt = Field(tokenize=tokenize_hindi, init_token="<sos>", eos_token="<eos>")

    # Defining Tabular Dataset
    data_fields = [('eng_text', english_txt), ('hindi_text', hindi_txt)]
    train_dt, val_dt = TabularDataset.splits(path='./', train='data/processed/train_sm.csv',
                                             validation='data/processed/val_sm.csv', format='csv', fields=data_fields)

    # Building word vocab
    english_txt.build_vocab(train_dt, max_size=10000, min_freq=2)
    hindi_txt.build_vocab(train_dt, max_size=10000, min_freq=2)

    # Training hyperparameters
    num_epochs = 1
    learning_rate = 3e-4
    batch_size = 256

    # Defining Iterator
    train_iter = BucketIterator(train_dt, batch_size=batch_size, sort_key=lambda x: len(x.eng_text), shuffle=True)
    val_iter = BucketIterator(val_dt, batch_size=batch_size, sort_key=lambda x: len(x.eng_text), shuffle=True)

    # Model hyper-parameters
    src_vocab_size = len(english_txt.vocab)
    trg_vocab_size = len(hindi_txt.vocab)
    embedding_size = 512
    num_heads = 8
    num_layers = 3
    dropout = 0.10
    max_len = 10000
    forward_expansion = 4
    src_pad_idx = english_txt.vocab.stoi["<pad>"]
    trg_pad_idx = 0

    # Defining model & optimizer attributes
    model = Transformer(src_vocab_size=src_vocab_size,
                        trg_vocab_size=trg_vocab_size,
                        src_pad_idx=src_pad_idx,
                        trg_pad_idx=trg_pad_idx,
                        embed_size=embedding_size,
                        num_layers=num_layers,
                        forward_expansion=forward_expansion,
                        heads=num_heads,
                        dropout=dropout,
                        device=device,
                        max_len=max_len).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    load_checkpoint(torch.load("nlp_models/my_checkpoint.pth.tar"), model, optimizer)

    src_field = english_txt
    src_tokenizer = tokenize_eng
    trg_field = hindi_txt
    trg_vcb_sz = 10000
    k = 5

    tr = beam_search(sentence=sentence, model=model, src_field=english_txt, src_tokenizer=tokenize_eng,
                     trg_field=hindi_txt, trg_vcb_sz=10000, k=5)
    return tr

