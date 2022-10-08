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

def cnn1d(timesteps, nions):
    inp = tf.keras.layers.Input(shape=(timesteps, nions+1))
    dilations = [1, 2, 5, 10, 20, 50]
    c = []
    for d in dilations:
        c_tmp = tf.keras.layers.Conv1D(8, 3, dilation_rate=d, padding='same')(inp)
        c_tmp = tf.keras.layers.BatchNormalization()(c_tmp)
        c_tmp = tf.keras.layers.Activation('relu')(c_tmp)
        c.append(c_tmp)

    c = tf.keras.layers.concatenate(c)
    x = tf.keras.layers.Flatten()(c)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    out = tf.keras.layers.Dense(9, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    return model

def resnet34(timesteps, nions):
    inp = tf.keras.layers.Input(shape=(timesteps, nions+1))

    x0 = tf.keras.layers.Reshape((timesteps, nions+1, 1))(inp)
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
