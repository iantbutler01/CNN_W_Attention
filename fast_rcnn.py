from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Embedding
from keras.optimizers import adam
import pickle
INPUT_SHAPE = (50, 300)
N_AUTHORS = 76
dicts = pickle.load('./dictionaries.p')
training_set = pickle.load('./train_dataset.p')

shared1 = Conv2D(filters=64, kernel_size=3, padding='same')
shared2 = Conv2D(filters=128, kernel_size=3, padding='same')
shared3 = Conv2D(filters=256, kernel_size=5, padding='same')
shared4 = Conv2D(filters=512, kernel_size=5, padding='same')

def rpn():
   out1 = shared1(output_last_shared)
   out2 = shared2(out1)
   mp_layer = MaxPooling2D()
   mp_out = mp_layer(out2)
   fc_layer = Dense(activation='relu')
   fc_out = fc_layer(mp)
   model = Model.compile(inputs=[output_last_shared], outputs=fc_out)
   def model_fn():
       return model
   return model_fn

def build_cnn_return_preds():
    inputs = Input(shape=INPUT_SHAPE)
    embed = Embedding(len(dicts['r_dict']), 128)
    shared1_out = shared1(inputs)
    shared2_out = shared2(shared1_out)
    mp1 = MaxPooling2D()
    mp1_out = mp1(shared2_out)
    fc1 = Dense(activation='relu')
    fc1_out = fc1(mp1_out)
    shared3_out = shared3(fc1_out)
    shared4_out = shared4(shared3_out)
    mp2 = MaxPooling2D()
    mp2_outs = mp2(shared4_out)
    fc_final = Dense(N_AUTHORS, activation='relu')
    fc_final_out = fc_final(mp2_outs)
    return fc_final_out

preds = build_cnn_return_preds(inputs)
model = Model(inputs=[], predictions=preds)
optimizer = adam(lr=0.003)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_input[0], train_input[1], epochs=100, batchsize=200)
model.save('./saved_model')
