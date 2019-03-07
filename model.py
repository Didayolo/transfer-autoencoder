import tensorflow as tf
import os
import numpy as np
from data_manager import *
Model = tf.keras.models.Model
Sequential = tf.keras.models.Sequential
Input, Dense, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Flatten, Reshape, Dropout = tf.keras.layers.Input, tf.keras.layers.Dense, tf.keras.layers.Conv2D, tf.keras.layers.Conv3D, tf.keras.layers.MaxPooling2D, tf.keras.layers.MaxPooling3D, tf.keras.layers.UpSampling2D, tf.keras.layers.UpSampling3D, tf.keras.layers.Flatten, tf.keras.layers.Reshape, tf.keras.layers.Dropout
regularizers = tf.keras.regularizers
load_model = tf.keras.models.load_model

MODELS_PATH = 'models'
DATA_PATH = '/home/adrien/Documents/autodl-data/image/formatted_datasets/'

class TAE():

    def __init__(self, name='autoencoder', shape=(1, 52, 52, 3), load=True, save=True, padding=False, train_test_split=True, verbose=False):
        """ Train autoencoder with data from different sources.
            Encode data and train a model.
        """
        self.load = load
        self.save = save
        self.verbose = verbose
        self.shape = shape
        self.padding = padding # Padding currently does not work for every dataset (TODO)
        self.train_test_split = train_test_split
        self.name = name
        self.path = os.path.join(MODELS_PATH, name+'.h5')
        self.save_path = self.path
        self.datasets = os.listdir(DATA_PATH)
        if load:
            self.autoencoder = self.load_autoencoder(self.path, verbose=verbose)
        else:
            self.autoencoder = self.init_autoencoder(name=name, input_shape=shape, verbose=verbose)
        self.encoder = self.init_encoder(verbose=verbose)
        self.decoder = self.init_decoder(verbose=verbose    )

    def init_autoencoder(self, name='autoencoder', input_shape=(1, 52, 52, 3), verbose=False):
        """ Definition of the autoencoder.
            TODO: automatically set architecture according to input_shape.
        """
        kernel = (1, 3, 3)
        pool = (1, 2, 2)
        strides = (2, 2, 2)
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
        """ Load autoencoder from file.
        """
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
        """ Save autoencoder to file.
        """
        self.autoencoder.save(path)
        name = os.path.basename(os.path.normpath(path))
        print('{} saved.'.format(name))

    def init_encoder(self, verbose=False):
        """ Return the first part of the autoencoder.
        """
        encoder = Model(inputs=self.autoencoder.input, outputs=self.autoencoder.get_layer('flatten', 5).output)
        print('encoder initialized.')
        if verbose:
            encoder.summary()
        return encoder

    def init_decoder(self, verbose=False):
        """ Return the second part of the autoencoder.
        """
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

    def init_model(self, output_dim=10, verbose=False, model=None):
        """ Define a model to solve a multiclass classification problem.
        """
        if model is None:
            model = Sequential()
            model.add(self.encoder)
            model.add(Dense(512, activation=tf.nn.relu))
            model.add(Dropout(0.2))
            model.add(Dense(output_dim, activation=tf.nn.sigmoid))
            # TODO: change metric to balanced accuracy?
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print('model initialized.')
            if verbose:
                model.summary()
        return model

    def metafit(self, datasets, meta_epoch=1, batch_size=256, epochs=1, steps_per_epoch=100, test_size=200):
        """
        Fit autoencoder on many datasets.
        Il serait interessant de tracer les courbes d'apprentissage sur les differents datasets simultanements.
        """
        for i in range(meta_epoch):
            print('Meta-epoch {}/{}'.format(i+1, meta_epoch))
            for dataset in datasets:
                print('Training on', dataset)
                x_train, y_train, x_test, y_test = load_dataset(DATA_PATH, dataset,
                                                                batch_size=batch_size,
                                                                shape=self.shape,
                                                                padding=self.padding,
                                                                train_test_split=self.train_test_split,
                                                                test_size=test_size)
                self.autoencoder.fit(x_train, x_train,
                                    epochs=epochs,
                                    steps_per_epoch=steps_per_epoch,
                                    validation_data=(x_test, x_test),
                                    validation_steps=steps_per_epoch)

        if self.save:
            self.save_autoencoder(self.save_path)

    def benchmark(self, datasets, batch_size=256, epochs=1, steps_per_epoch=100, test_size=200, model=None):
        """ Train and evaluate model on datasets.
        """
        for dataset in datasets:
            print('Testing on', dataset)
            x_train, y_train, x_test, y_test = load_dataset(DATA_PATH, dataset,
                                                            batch_size=batch_size,
                                                            shape=self.shape,
                                                            padding=self.padding,
                                                            train_test_split=self.train_test_split,
                                                            test_size=test_size)
            with tf.Session() as sess:
                tensor_4d, labels = sess.run(x_train), sess.run(y_train)
                output_dim = labels[0].shape[0]
                print('Output dimension:', output_dim)

            if model is None:
                model = self.init_model(output_dim=output_dim)
                model.fit(x_train, y_train, epochs=epochs, steps_per_epoch=steps_per_epoch)
                print(model.evaluate(x_test, y_test, steps=steps_per_epoch))
            else:
                test_steps = int(np.ceil(test_size / batch_size))
                # extract features
                features_train = self.encoder.predict(x_train, steps=steps_per_epoch)
                features_test = self.encoder.predict(x_test, steps=test_steps)
                model.fit(features_train, tensor_to_array(y_train))
                model.score(features_test, tensor_to_array(y_test))

    def show_example(self, dataset, n=0, test=True):
        tensor_4d, labels = get_example(DATA_PATH, dataset, shape=self.shape, padding=self.padding, train_test_split=self.train_test_split, test=test)
        show_example(tensor_4d[n], labels[n])

    def show_reconstructed_example(self, dataset, n=0, test=True):
        tensor_4d, labels = get_example(DATA_PATH, dataset, shape=self.shape, padding=self.padding, train_test_split=self.train_test_split, test=test)
        reconstructed_tensor_4d = self.autoencoder.predict(tensor_4d)
        show_example(reconstructed_tensor_4d[n])


    # TODO
    # self.model = self.init_model(verbose=verbose)
    #def fit(self):
    #    return self.model.fit
    #def predict(self):
    #    return self.model.predict
