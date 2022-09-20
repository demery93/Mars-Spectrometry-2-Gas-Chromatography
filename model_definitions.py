import tensorflow as tf
import keras
import tfimm

def cnn(timesteps, nions):
    inp = tf.keras.layers.Input(shape=(timesteps, nions))

    c1 = tf.keras.layers.Conv1D(50, 2, strides=1, dilation_rate=2 ** 0, padding='same')(inp)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Activation('relu')(c1)
    c2 = tf.keras.layers.Conv1D(50, 2, strides=1, dilation_rate=2 ** 1, padding='same')(inp)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Activation('relu')(c2)
    c3 = tf.keras.layers.Conv1D(50, 2, strides=1, dilation_rate=2 ** 2, padding='same')(inp)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Activation('relu')(c3)
    c4 = tf.keras.layers.Conv1D(50, 2, strides=1, dilation_rate=2 ** 3, padding='same')(inp)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Activation('relu')(c4)
    c5 = tf.keras.layers.Conv1D(50, 2, strides=1, dilation_rate=2 ** 4, padding='same')(inp)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Activation('relu')(c5)
    c = tf.keras.layers.concatenate([c1, c2, c3, c4, c5])

    x = tf.keras.layers.Flatten()(c)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    out = tf.keras.layers.Dense(9, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def lstm(timesteps, nions):
    inp = tf.keras.layers.Input(shape=(timesteps, nions))

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=False))(inp)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

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

def resnet34(timesteps, nions):
    inp = tf.keras.layers.Input(shape=(timesteps, nions))

    x0 = tf.keras.layers.Reshape((timesteps, nions, 1))(inp)

    m = tfimm.create_model("resnet34", pretrained=True, in_channels=1)
    res, features = m(x0, return_features=True)
    x = features['features']

    p1 = tf.keras.layers.GlobalAveragePooling2D()(x)
    p2 = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.concatenate([p1, p2])
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    out = tf.keras.layers.Dense(9, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    return model
