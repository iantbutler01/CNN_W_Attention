from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling1D, Input, Embedding, Conv1D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import pickle
import numpy as np
INPUT_SHAPE = (40,)
N_AUTHORS = 3
dicts = pickle.load(open('./dictionaries.p', 'rb'))

def conv_blocks(inputs, n=2):
    _inputs = inputs
    for i in range(0, n):
        conv1 = Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')
        conv2 = Conv1D(filters=512, kernel_size=5, padding='same', activation='relu')
        mp1 = MaxPooling1D(pool_size=(32,), padding='same')
        mp2 = MaxPooling1D(pool_size=(2,), padding='same')
        conv1_out = conv1(_inputs)
        mp1_out = mp1(conv1_out)
        conv2_out = conv2(mp1_out)
        mp2_out = mp2(conv2_out)
        dropout = Dropout(0.8)
        _inputs = dropout(mp2_out)
    return _inputs

def build_cnn_return_preds(inputs, n_layers):
    embed = Embedding(len(dicts['r_dict']), 40, input_length=40)
    embedded_inputs = embed(inputs)
    conv_outs = conv_blocks(embedded_inputs, n_layers)
    fc_final = Dense(N_AUTHORS, activation='softmax')
    fc_final_out = fc_final(conv_outs)
    return (fc_final_out, None)
