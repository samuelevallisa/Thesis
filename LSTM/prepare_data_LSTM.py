import numpy as np
import torch 
from torch.utils.data import DataLoader, Dataset, Sampler

def data_LSTM(enc_in,dec_in,label,batch_size):
    enc_in_lstm = np.transpose(enc_in,axes=(2,3,4,0,1))
    dec_in_lstm = np.transpose(dec_in,axes=(2,3,4,0,1))
    label_lst = np.transpose(label,axes=(2,3,4,0,1))

    num_features_enc = enc_in_lstm.shape[2] 
    num_features_dec = dec_in_lstm.shape[2]

    enc_in_lstm = torch.from_numpy(enc_in_lstm).float()
    dec_in_lstm = torch.from_numpy(dec_in_lstm).float()
    label_lstm =  torch.from_numpy(label_lst).float()

    train_set_lstm = train_set = torch.utils.data.TensorDataset(enc_in_lstm, dec_in_lstm, label_lstm)
    sampler_lstm  = torch.utils.data.SequentialSampler(train_set_lstm)
    train_loader_lstm = torch.utils.data.DataLoader(train_set_lstm, batch_size = batch_size,shuffle=False,sampler=sampler_lstm)

    return train_loader_lstm, num_features_enc, num_features_dec

def data_LSTM_test(enc_in,dec_in,label,maximum,minimum,batch_size):
    encoder_input = np.transpose(enc_in,axes=(2,3,4,0,1))
    decoder_input = np.transpose(dec_in,axes=(2,3,4,0,1))
    label_input = np.transpose(label,axes=(2,3,4,0,1))

    enc_input = torch.from_numpy(encoder_input).float()
    dec_input = torch.from_numpy(decoder_input).float()
    label_input =  torch.from_numpy(label_input).float()
    maximum=torch.from_numpy(maximum).float()
    minimum=torch.from_numpy(minimum).float()

    train_set = torch.utils.data.TensorDataset(enc_input, dec_input, label_input)
    sampler_lstm  = torch.utils.data.SequentialSampler(train_set)
    train_loader_lstm = torch.utils.data.DataLoader(train_set, batch_size = batch_size,shuffle=False,sampler=sampler_lstm)

    return train_loader_lstm,maximum,minimum