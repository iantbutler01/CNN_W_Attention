from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling1D, Input, Embedding, Conv1D, AveragePooling1D, Flatten, Dot, Multiply, Activation
from keras.layers import Concatenate, Add, Lambda
from keras.optimizers import adam
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy
from keras import backend as K
from keras.callbacks import EarlyStopping
import pickle
import numpy as np
INPUT_SHAPE = (400,)
N_AUTHORS = 70
dicts = pickle.load(open('./dictionaries.p', 'rb'))
training_set = pickle.load(open('./train_dataset.p', 'rb'))

shared1 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
shared2 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
shared3 = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')
shared4 = Conv1D(filters=128, kernel_size=6, padding='same', activation='relu')

def attention_layer(inputs, filters):
    #Compute Feature Maps
    M = Conv1D(filters=filters, kernel_size=1, padding='same', activation='softmax')
    M_out = M(inputs)
    mp = Multiply()
    Z = mp([inputs, M_out])
    #Dimension Reduction
    reduce_dims = AveragePooling1D(pool_size=2, strides=2, padding='same')
    reduce_dims_out = reduce_dims(Z)
    #Attention Class Prediction
    fc = Dense(N_AUTHORS, activation='softmax')
    _class = fc(reduce_dims_out)
    #Calculate Confidence
    confidence_layer = Dense(N_AUTHORS, activation='sigmoid')
    c_layer_out = confidence_layer(reduce_dims_out)
    confidence = Lambda(lambda x: K.sum(x, axis=0))([c_layer_out])
    return (M_out, Z, _class, confidence)

def frobinius_regulzarization(maps):
    norms = []
    for x in range(1, len(maps)):
        norms.append(K.sqrt(K.sum((maps[x]-maps[x-1])**2)))
    return K.sum(norms)

def losses(maps):
    def combined_loss(y_true, y_pred):
        loss1 = categorical_crossentropy(y_true, y_pred)
        loss2 = frobinius_regulzarization(maps)
        return loss1 + 0.1*loss2
    return combined_loss

def calculate_attention_weight(confidences, att_preds):
    softmaxes = []
    for x,y in zip(confidences, att_preds):
        softmaxes.append(Lambda(lambda x: K.softmax(x[0]*x[1]))([x, y]))
    return Add()(softmaxes)

def build_cnn_return_preds(inputs):
    attn_maps = []
    attn_confidences = []
    attn_preds = []
    # Embed the words
    embed = Embedding(len(dicts['r_dict']), 128, input_length=400)
    embedded_inputs = embed(inputs)
    #conv block 1
    shared1_out = shared1(embedded_inputs)
    shared2_out = shared2(shared1_out)
    mp1 = MaxPooling1D(pool_size=(5,), padding='same')
    mp1_out = mp1(shared2_out)
    #attn block 1
    attention1 = attention_layer(mp1_out, 128)
    attn_maps.append(attention1[0])
    attn_confidences.append(attention1[-1])
    attn_preds.append(attention1[-2])
    #conv block 2
    shared3_out = shared3(mp1_out)
    shared4_out = shared4(shared3_out)
    mp2 = MaxPooling1D(pool_size=(50,))
    mp2_outs = mp2(shared4_out)
    #attn block 2
    attention2 = attention_layer(mp2_outs, 128)
    attn_maps.append(attention2[0])
    attn_confidences.append(attention2[-1])
    attn_preds.append(attention2[-2])
    #predictions
    fc_final = Dense(N_AUTHORS, activation='softmax')
    fc_final_out = fc_final(mp2_outs)
    #network without attention confidence score
    confidence_layer = Dense(N_AUTHORS, activation='sigmoid')
    c_out = Lambda(lambda x: K.sum(x, axis=0))(confidence_layer(mp2_outs))
    #get how much each attention head should contribute
    attn_attributions = calculate_attention_weight(attn_confidences, attn_preds)
    #apply the confidence to the predictions
    gated_out = Multiply()([c_out, fc_final_out])
    #add the contributions from the attention heads
    attention_weighted_out = Add()([gated_out, attn_attributions])
    return (attention_weighted_out, attn_maps)

inputs = Input(shape=INPUT_SHAPE)
preds, maps = build_cnn_return_preds(inputs)
model = Model(inputs=[inputs], outputs=preds)
optimizer = adam(lr=0.003)
model.compile(optimizer=optimizer, loss=losses(maps), metrics=['accuracy'])
inputs = []
labels = []
for _, v in training_set.items():
    for x in v:
        inputs.append(x['text'])
        labels.append(x['author'])
inputs = np.asarray(inputs)
labels = np.expand_dims(to_categorical(np.asarray(labels)), 1)
model.fit(inputs, labels, epochs=20, batch_size=100, shuffle='batch')
model.save('./saved_model')
