import tensorflow as tf
import os
Model = tf.keras.models.Model
Sequential = tf.keras.models.Sequential
Input, Dense, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Flatten, Reshape, Dropout = tf.keras.layers.Input, tf.keras.layers.Dense, tf.keras.layers.Conv2D, tf.keras.layers.Conv3D, tf.keras.layers.MaxPooling2D, tf.keras.layers.MaxPooling3D, tf.keras.layers.UpSampling2D, tf.keras.layers.UpSampling3D, tf.keras.layers.Flatten, tf.keras.layers.Reshape, tf.keras.layers.Dropout
regularizers = tf.keras.regularizers
load_model = tf.keras.models.load_model

class TAE():

    def __init__(self, autoencoder_name='autoencoder', shape=(1, 52, 52, 3), load=True, path='', verbose=False):
        if load:
            self.autoencoder = self.load_autoencoder(path, verbose=verbose)
        else:
            self.autoencoder = self.init_autoencoder(name=autoencoder_name, input_shape=shape, verbose=verbose)

        self.encoder = self.init_encoder(verbose=verbose)
        self.model = self.init_model(verbose=verbose)

    def init_autoencoder(self, name='autoencoder', input_shape=(1, 52, 52, 3), verbose=False):
        # TODO: automatically set architecture according to input_shape
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
        autoencoder.add(Reshape((1, 7, 7, 8)))

        # Decoder Layers
        autoencoder.add(Conv3D(8, kernel, activation='relu', padding='same'))
        autoencoder.add(UpSampling3D(pool))
        autoencoder.add(Conv3D(8, kernel, activation='relu', padding='same'))
        autoencoder.add(UpSampling3D(pool))
        autoencoder.add(Conv3D(16, kernel, activation='relu'))
        autoencoder.add(UpSampling3D(pool))
        autoencoder.add(Conv3D(3, kernel, activation='sigmoid', padding='same'))

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

    def init_decoder(self, verbose=False):
        # TODO : construct automatically from input_shape
        #decoder = Model(inputs=self.autoencoder.get_layer('reshape', 6), outputs=self.autoencoder.get_layer('conv_3d', -1).output)
        encoded_input = Input(shape=(392,))
        deco = self.autoencoder.layers[-8](encoded_input)
        for i in range(-7, 0):
            deco = self.autoencoder.layers[i](deco)
        # create the decoder model
        decoder = Model(encoded_input, deco)
        print('decoder initialized.')
        if verbose:
            decoder.summary()
        return decoder

    def init_model(self, output_dim=10, verbose=False):
        model = Sequential()
        model.add(self.encoder)
        model.add(Dense(512, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim, activation=tf.nn.sigmoid))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print('model initialized.')
        if verbose:
            model.summary()
        return model
