import tensorflow as tf
import os
Model = tf.keras.models.Model
Sequential = tf.keras.models.Sequential
Input, Dense, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Flatten, Reshape, Dropout = tf.keras.layers.Input, tf.keras.layers.Dense, tf.keras.layers.Conv2D, tf.keras.layers.Conv3D, tf.keras.layers.MaxPooling2D, tf.keras.layers.MaxPooling3D, tf.keras.layers.UpSampling2D, tf.keras.layers.UpSampling3D, tf.keras.layers.Flatten, tf.keras.layers.Reshape, tf.keras.layers.Dropout
regularizers = tf.keras.regularizers
load_model = tf.keras.models.load_model

class TAE():

    def __init__(self, autoencoder_name='autoencoder', load=True, path='', verbose=False):
        if load:
            self.autoencoder = self.load_autoencoder(path, verbose=verbose)
        else:
            self.autoencoder = self.init_autoencoder(name=autoencoder_name, verbose=verbose)

        self.encoder = self.init_encoder(verbose=verbose)
        self.model = self.init_model(verbose=verbose)

    def init_autoencoder(self, name='autoencoder', input_shape=(1,28,28,1), verbose=False):
        kernel = (1, 3, 3)
        pool = (1, 2, 2)
        strides = (2,2,2)
        autoencoder = Sequential()

        # Encoder Layers
        autoencoder.add(Conv3D(16, kernel, activation='relu', padding='same', input_shape=input_shape))
        autoencoder.add(MaxPooling3D(pool, padding='same'))
        autoencoder.add(Conv3D(8, kernel, activation='relu', padding='same'))
        autoencoder.add(MaxPooling3D(pool, padding='same'))
        autoencoder.add(Conv3D(8, kernel, strides=strides, activation='relu', padding='same'))

        # Flatten encoding for visualization
        autoencoder.add(Flatten())
        autoencoder.add(Reshape((1, 4, 4, 8)))

        # Decoder Layers
        autoencoder.add(Conv3D(8, kernel, activation='relu', padding='same'))
        autoencoder.add(UpSampling3D(pool))
        autoencoder.add(Conv3D(8, kernel, activation='relu', padding='same'))
        autoencoder.add(UpSampling3D(pool))
        autoencoder.add(Conv3D(16, kernel, activation='relu'))
        autoencoder.add(UpSampling3D(pool))
        autoencoder.add(Conv3D(1, kernel, activation='sigmoid', padding='same'))

        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        print('{} initialized.'.format(name))
        if verbose:
            autoencoder.summary()
        return autoencoder

    def load_autoencoder(self, path, verbose=False):
        if os.path.isfile(path):
            autoencoder = load_model(path)
            autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
            model_name = os.path.basename(os.path.normpath(path))
            print('{} loaded.'.format(model_name))
            return autoencoder
        else:
            print('autoencoder file not found: {}'.format(path))
            return self.init_autoencoder(verbose=verbose)

    def save_autoencoder(self, path, verbose=False):
        self.autoencoder.save(path)
        name = os.path.basename(os.path.normpath(path))
        print('{} saved.'.format(name))

    def init_encoder(self, verbose=False):
        encoder = Model(inputs=self.autoencoder.input, outputs=self.autoencoder.get_layer('flatten', 5).output)
        print('encoder initialized.')
        if verbose:
            encoder.summary()
        return encoder

    def init_model(self, verbose=False):
        model = Sequential()
        model.add(self.encoder)
        model.add(Dense(512, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation=tf.nn.sigmoid))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print('model initialized.')

        if verbose:
            model.summary()

        return model
