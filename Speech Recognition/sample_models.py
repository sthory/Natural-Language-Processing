from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout, MaxPooling1D)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name = 'the_input', 
                       shape = (None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, 
                     kernel_size, 
                     strides = conv_stride, 
                     padding = conv_border_mode,
                     activation = 'relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name = 'bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, 
                         activation = 'relu',
                         return_sequences = True, 
                         implementation = 2, 
                         name = 'rnn')(bn_cnn)
    
    # Add batch normalization
    bn_rnn = BatchNormalization(name = 'bn_rnn') (simp_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', 
                        name = 'softmax')(time_dense)
    # Specify the model
    model = Model(inputs = input_data, 
                  outputs = y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation = 1, pooling = 1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // (stride * pooling)


def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', 
                       shape = (None, input_dim))

    # Add recurrent layers, each with batch normalization
    previous_layer = input_data
    for i in range(recur_layers):
        previous_layer = GRU(units, 
                             activation = 'relu',
                             return_sequences = True, 
                             implementation = 2, 
                             name = f'rnn_{i+1}')(previous_layer)
        previous_layer = BatchNormalization()(previous_layer)
    
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(previous_layer)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', 
                        name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs = input_data, 
                  outputs = y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', 
                       shape=(None, input_dim))
    
    # Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, 
                                  activation = 'relu',
                                  return_sequences = True,
                                  implementation = 2,
                                  name = 'bidir_rnn'),
                                  merge_mode = 'concat')(input_data)
    
    bidir_rnn = BatchNormalization()(bidir_rnn)
    
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    
    
    # Add softmax activation layer
    y_pred = Activation('softmax', 
                        name = 'softmax')(time_dense)
    # Specify the model
    model = Model(inputs = input_data, 
                  outputs = y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# My model
def cnn_bi_rnn_max_drop(input_dim, filters, kernel_size, 
                        conv_stride, units, conv_border_mode, 
                        cnn_qtty, output_dim, dropout, pool_size):
    
    """ RNN + CNN x (n) + Bidirectional + maxpooling + dropout
    """
    # Main acoustic input
    input_data = Input(name = 'the_input', 
                       shape = (None, input_dim))
    
    # add cnn x (n)
    previous_layer = input_data
    for i in range(cnn_qtty):
        # Add convolutional layer
        conv_1d = Conv1D(filters, 
                         kernel_size, 
                         strides = conv_stride, 
                         padding = conv_border_mode,
                         activation = 'relu',
                         name = f'conv1d_{i+1}')(previous_layer)
        
        # batch normalization
        bn_cnn = BatchNormalization(name = f'bn_conv_1d_{i+1}')(conv_1d)
        
        # dropout 
        previous_layer = Dropout(dropout, 
                                 name = f'cnn_dropout_{i+1}')(bn_cnn)
        
    # max pooling 
    pool_cnn = MaxPooling1D(pool_size, 
                            name = f'cnn_max_pool_{i+1}')(previous_layer)
    
    # Add a recurrent layer
    rnn = GRU(units, 
              activation = 'relu', 
              dropout = dropout, 
              return_sequences = True, 
              implementation = 2, 
              name = 'rnn')
    
    bidir_rnn = Bidirectional(rnn, 
                              merge_mode = 'concat')(pool_cnn)
    
    # batch normalization
    bn_rnn = BatchNormalization(name = 'bn_rnn')(bidir_rnn)
    
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', 
                        name = 'softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs = input_data, 
                  outputs = y_pred)
    
    model.output_length = lambda x: cnn_output_length(x, 
                                                      kernel_size, 
                                                      conv_border_mode, 
                                                      conv_stride,  
                                                      pooling = pool_size)
    
    print(model.summary())
    return model


def final_model(input_dim, filters, kernel_size, 
                conv_stride, units, conv_border_mode, 
                cnn_qtty, rnn_qtty, output_dim, 
                dropout, pool_size):
    
    """ RNN + CNN x (n) + Bidirectional x (n) + maxpooling + dropout +
        + TimeDistributed
    """
    # Build a deep network for speech 
    
    # Main acoustic input
    input_data = Input(name = 'the_input', 
                       shape = (None, input_dim))
        
    # Specify the layers in your network
    
    # add cnn x (n) layers 
    previous_layer = input_data
    for n in range(cnn_qtty):
        # Add convolutional layer
        conv_1d = Conv1D(filters, 
                         kernel_size, 
                         strides = conv_stride, 
                         padding = conv_border_mode,
                         activation = 'relu',
                         name = f'conv1d_{n+1}')(previous_layer)
        
        # batch normalization
        bn_cnn = BatchNormalization(name = f'bn_conv_1d_{n+1}')(conv_1d)
        
        # dropout 
        previous_layer = Dropout(dropout, name = f'cnn_dropout_{n+1}')(bn_cnn)
    
    # max pooling 
    pool_cnn = MaxPooling1D(pool_size, 
                            name=f'cnn_max_pool_{n+1}')(previous_layer)
    
    # Add a rnn x (n) layers
    
    previous_layer = pool_cnn
    for n in range(rnn_qtty):
        
        # define rnn
        rnn = GRU(units, 
                  activation = 'relu', 
                  dropout = dropout, 
                  return_sequences = True, 
                  implementation = 2, 
                  name = f'rnn_{n+1}')
    
        # add Bidirectional
        bidir_rnn = Bidirectional(rnn, 
                                  merge_mode = 'concat')(previous_layer)
    
        # batch normalization
        previous_layer = BatchNormalization(name = f'bn_rnn_{n+1}')(bidir_rnn)
    
    # add TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(previous_layer)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', 
                        name = 'softmax')(time_dense)

    # Specify the model
    model = Model(inputs = input_data, 
                  outputs = y_pred)
    
    # Specify model.output_length
    model.output_length = lambda x: cnn_output_length(x, 
                                                      kernel_size, 
                                                      conv_border_mode, 
                                                      conv_stride,  
                                                      pooling = pool_size)
    print(model.summary())
    return model
