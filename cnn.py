from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling1D, Input, Embedding, Conv1D, MaxPooling2D, Flatten
from keras.optimizers import adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import pickle
import numpy as np
INPUT_SHAPE = (400,)
N_AUTHORS = 70
dicts = pickle.load(open('./dictionaries.p', 'rb'))
training_set = pickle.load(open('./train_dataset.p', 'rb'))

# shared1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
# shared2 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')
# shared3 = Conv2D(filters=256, kernel_size=5, padding='same', activation='relu')
# shared4 = Conv2D(filters=512, kernel_size=5, padding='same', activation='relu')

shared1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
shared2 = Conv1D(filters=64, kernel_size=4, padding='same', activation='relu')
shared3 = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')
shared4 = Conv1D(filters=128, kernel_size=6, padding='same', activation='relu')

def build_cnn_return_preds(inputs):
    embed = Embedding(len(dicts['r_dict']), 128, input_length=400)
    embedded_inputs = embed(inputs)
    print(embedded_inputs.shape)
    shared1_out = shared1(embedded_inputs)
    print(shared1_out.shape)
    shared2_out = shared2(shared1_out)
    print(shared2_out.shape)
    mp1 = MaxPooling1D(pool_size=(5,), padding='same')
    mp1_out = mp1(shared2_out)
    print(mp1_out.shape)
    shared3_out = shared3(mp1_out)
    print(shared3_out.shape)
    shared4_out = shared4(shared3_out)
    print(shared4_out.shape)
    mp2 = MaxPooling1D(pool_size=(50,))
    mp2_outs = mp2(shared4_out)
    print(mp2_outs.shape)
    fc_final = Dense(N_AUTHORS, activation='softmax')
    fc_final_out = fc_final(mp2_outs)
    print(fc_final_out.shape)
    return fc_final_out

inputs = Input(shape=INPUT_SHAPE)
preds = build_cnn_return_preds(inputs)
model = Model(inputs=[inputs], outputs=preds)
optimizer = adam(lr=0.003)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
inputs = []
labels = []
for _, v in training_set.items():
    for x in v:
        inputs.append(x['text'])
        labels.append(x['author'])
inputs = np.asarray(inputs)
labels = np.expand_dims(to_categorical(np.asarray(labels)), 1)
e_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=0, mode='auto')
model.fit(inputs, labels, epochs=20, batch_size=100, shuffle='batch', callbacks=[e_stop])
model.save('./saved_model')
