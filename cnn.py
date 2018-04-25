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

shared1 = Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')
shared2 = Conv1D(filters=512, kernel_size=5, padding='same', activation='relu')

def build_cnn_return_preds(inputs):
    embed = Embedding(len(dicts['r_dict']), 40, input_length=40)
    embedded_inputs = embed(inputs)
    shared1_out = shared1(embedded_inputs)
    mp1 = MaxPooling1D(pool_size=(32,), padding='same')
    mp1_out = mp1(shared1_out)
    shared2_out = shared2(mp1_out)
    mp2 = MaxPooling1D(pool_size=(2,), padding='same')
    mp2_out = mp2(shared2_out)
    dropout = Dropout(0.8)
    dropout_outs = dropout(mp2_out)
    fc_final = Dense(N_AUTHORS, activation='softmax')
    fc_final_out = fc_final(dropout_outs)
    return (fc_final_out, None)
