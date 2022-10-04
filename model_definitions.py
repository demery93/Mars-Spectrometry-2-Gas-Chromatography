import tensorflow as tf
import keras
import tfimm
from einops.layers.keras import Rearrange

def lstm(timesteps, nions):
    inp = tf.keras.layers.Input(shape=(timesteps, nions))

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=False))(inp)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(9, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    return model

def cnn2d(timesteps, nions):
    inp = tf.keras.layers.Input(shape=(timesteps, nions))

    x0 = tf.keras.layers.Reshape((timesteps, nions, 1))(inp)

    m = tfimm.create_model("resnet34", pretrained=True, in_channels=1, features_only=True)
    res, features = m(x0, return_features=True)
    x = tf.keras.layers.Dense(128, activation='relu')(features)

    out = tf.keras.layers.Dense(9, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def cnn1d(timesteps, nions):
    inp = tf.keras.layers.Input(shape=(timesteps, nions))

    c1 = tf.keras.layers.Conv1D(50, 3, dilation_rate=2 ** 0, padding='same')(inp)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Activation('relu')(c1)
    c2 = tf.keras.layers.Conv1D(50, 3, dilation_rate=2 ** 1, padding='same')(inp)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Activation('relu')(c2)
    c3 = tf.keras.layers.Conv1D(50, 3, dilation_rate=2 ** 2, padding='same')(inp)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Activation('relu')(c3)
    c4 = tf.keras.layers.Conv1D(50, 3, dilation_rate=2 ** 3, padding='same')(inp)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Activation('relu')(c4)
    c5 = tf.keras.layers.Conv1D(50, 3, dilation_rate=2 ** 4, padding='same')(inp)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Activation('relu')(c5)

    c = tf.keras.layers.concatenate([c1, c2, c3, c4, c5])
    x = tf.keras.layers.Flatten()(c)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    out = tf.keras.layers.Dense(9, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    return model

def resnet34(timesteps, nions):
    inp = tf.keras.layers.Input(shape=(timesteps, nions))

    x0 = tf.keras.layers.Reshape((timesteps, nions, 1))(inp)
    x0 = Rearrange("b t n c -> b n t c")(x0)

    m = tfimm.create_model("resnet34", in_channels=1, pretrained=True)
    res, features = m(x0, return_features=True)
    x = features['features']

    p1 = tf.keras.layers.AveragePooling2D(x.shape[1:3])(x)
    p2 = tf.keras.layers.MaxPooling2D(x.shape[1:3])(x)

    x = tf.keras.layers.concatenate([p1, p2])
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    out = tf.keras.layers.Dense(9, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    return model

def SimpleCls3(timesteps, nions):
    inp = tf.keras.layers.Input(shape=(timesteps, nions, 3))

    m = tfimm.create_model("resnet34", pretrained=True, in_channels=3, nb_classes=9)
    out = m(inp)
    '''x = features['features']

    p1 = tf.keras.layers.GlobalAveragePooling2D()(x)
    p2 = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.concatenate([p1, p2])
    x = tf.keras.layers.Dense(128, activation='relu')(x)'''

    #out = tf.keras.layers.Dense(9, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    return model


def cbr(x, out_layer, kernel, stride, dilation):
    x = tf.keras.layers.Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def wave_block(x, filters, kernel_size, n):
    dilation_rates = [2 ** i for i in range(n)]
    x = tf.keras.layers.Conv1D(filters=filters,
               kernel_size=1,
               padding='same')(x)
    res_x = x
    for dilation_rate in dilation_rates:
        tanh_out = tf.keras.layers.Conv1D(filters=filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='tanh',
                          dilation_rate=dilation_rate)(x)
        sigm_out = tf.keras.layers.Conv1D(filters=filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='sigmoid',
                          dilation_rate=dilation_rate)(x)
        x = tf.keras.layers.Multiply()([tanh_out, sigm_out])
        x = tf.keras.layers.Conv1D(filters=filters,
                   kernel_size=1,
                   padding='same')(x)
        res_x = tf.keras.layers.Add()([res_x, x])
    return res_x

def cnn(timesteps, nions, kernel_width=3, input_smoothing=20):
    abundance_in = tf.keras.layers.Input(shape=(timesteps, nions))
    time_in = tf.keras.layers.Input(shape=(timesteps, 1))

    x_in = tf.keras.layers.concatenate([abundance_in, time_in], axis=2)
    x_in = tf.keras.layers.Conv1D(128, input_smoothing, strides=input_smoothing)(x_in)

    x = cbr(x_in, 32, kernel_width, 1, 1)
    x = cbr(x, 32, kernel_width, 1, 2)
    x = cbr(x, 32, kernel_width, 1, 4)
    x = cbr(x, 32, kernel_width, 1, 8)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    out = tf.keras.layers.Dense(9, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=[abundance_in, time_in], outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model
