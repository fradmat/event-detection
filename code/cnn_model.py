import numpy as np
from keras.layers import Conv1D, Input, MaxPooling1D, Flatten, concatenate, Dense, Dropout, BatchNormalization, Reshape
from keras.models import Model
from keras.callbacks import Callback

def cnn(conv_w_size=41, conv_channels=3):
    conv_input = Input(shape=(conv_w_size,conv_channels,), dtype='float32', name='conv_input')
    
    x_conv = Conv1D(64, 3, activation='relu', padding='same')(conv_input)
    x_conv = Conv1D(128, 3, activation='relu', padding='same')(x_conv)
    x_conv = Dropout(.5)(x_conv)
    x_conv = MaxPooling1D(2)(x_conv)
    x_conv = Conv1D(256, 3, activation='relu', padding='same')(x_conv)
    x_conv = Conv1D(256, 3, activation='relu', padding='same')(x_conv)
    x_conv = Conv1D(256, 3, activation='relu', padding='same')(x_conv)
    x_conv = Dropout(.5)(x_conv)
    x_conv = MaxPooling1D(2)(x_conv)
    
    x_conv = Conv1D(256, 3, activation='relu', padding='same')(x_conv)
    x_conv = Conv1D(256, 3, activation='relu', padding='same')(x_conv)
    x_conv = Conv1D(256, 3, activation='relu', padding='same')(x_conv)
    x_conv = Dropout(.5)(x_conv)
    x_conv = MaxPooling1D(2)(x_conv)
    
    
    conv_out = Flatten()(x_conv)
    conv_out = Dense(64, activation='relu', name='scalar_output')(conv_out)
    conv_out = Dense(16, activation='relu')(conv_out)
    out_elms = Dense(2, activation='softmax', name='out_elms')(conv_out)
    out_transitions = Dense(7, activation='softmax', name='out_transitions')(conv_out)
    # x_dithers = Dense(64, activation='relu')(x_all)
    # out_dithers = Dense(2, activation='softmax', name='out_dithers')(x_dithers)
    model = Model(inputs=[conv_input], outputs=[out_elms, out_transitions])
    # model = Model(inputs=[conv_input], outputs=[out_elms, out_transitions, out_dithers])
    # model = Model(inputs=[conv_input], outputs=[out_transitions])
    print(model.summary())
    return model

