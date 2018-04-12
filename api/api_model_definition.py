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
import math
def attention_layer(inputs, filters, n_authors):
    #Compute Feature Maps
    M = Conv1D(filters=filters, kernel_size=1, padding='same', activation='softmax')
    M_out = M(inputs)
    mp = Multiply()
    Z = mp([inputs, M_out])
    #Dimension Reduction
    reduce_dims = AveragePooling1D(pool_size=2, strides=2, padding='same')
    reduce_dims_out = reduce_dims(Z)
    #Attention Class Prediction
    fc = Dense(n_authors, activation='softmax')
    _class = fc(reduce_dims_out)
    #Calculate Confidence
    confidence_layer = Dense(n_authors, activation='sigmoid')
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

def build_cnn_return_preds(input_shape, n_authors, vocab_size):
    #define conv layers
    inputs = Input(shape=input_shape)
    conv1 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=6, padding='same', activation='relu')
    attn_maps = []
    attn_confidences = []
    attn_preds = []
    # Embed the words
    embed = Embedding(vocab_size, 128, input_length=400)
    embedded_inputs = embed(inputs)
    #conv block 1
    conv1_out = conv1(embedded_inputs)
    conv2_out = conv2(conv1_out)
    mp1 = MaxPooling1D(pool_size=(5,), padding='same')
    mp1_out = mp1(conv2_out)
    #attn block 1
    attention1 = attention_layer(mp1_out, 128, n_authors)
    attn_maps.append(attention1[0])
    attn_confidences.append(attention1[-1])
    attn_preds.append(attention1[-2])
    #conv block 2
    conv3_out = conv3(mp1_out)
    conv4_out = conv4(conv3_out)
    pool_size = int(math.ceil(input_shape[0]/5))
    mp2 = MaxPooling1D(pool_size=(pool_size,))
    mp2_outs = mp2(conv4_out)
    #attn block 2
    attention2 = attention_layer(mp2_outs, 128, n_authors)
    attn_maps.append(attention2[0])
    attn_confidences.append(attention2[-1])
    attn_preds.append(attention2[-2])
    #predictions
    fc_final = Dense(n_authors, activation='softmax')
    fc_final_out = fc_final(mp2_outs)
    #network without attention confidence score
    confidence_layer = Dense(n_authors, activation='sigmoid')
    c_out = Lambda(lambda x: K.sum(x, axis=0))(confidence_layer(mp2_outs))
    #get how much each attention head should contribute
    attn_attributions = calculate_attention_weight(attn_confidences, attn_preds)
    #apply the confidence to the predictions
    gated_out = Multiply()([c_out, fc_final_out])
    #add the contributions from the attention heads
    attention_weighted_out = Add()([gated_out, attn_attributions])
    model = Model(inputs=[inputs], outputs=attention_weighted_out)
    optimizer = adam(lr=0.003)
    model.compile(optimizer=optimizer, loss=losses(attn_maps), metrics=['accuracy'])
    return (model, attn_maps)


