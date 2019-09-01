#!/usr/bin/env python3
""" GAN Model """

import numpy as np
from keras.initializers import RandomNormal
from keras.layers import add, Dense, Flatten, Input, Reshape, Conv2D
from keras.layers.advanced_activations import LeakyReLU

from keras.models import Model as KerasModel

from lib.model.layers import PixelShuffler
from ._base import ModelBase, logger

class Model(ModelBase):
    """ GAN Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        self.configfile = kwargs.get("configfile", None)
        kwargs["input_shape"] = (128, 128, 3)
        kwargs["encoder_dim"] = 512 if self.config["lowmem"] else 1024
        self.kernel_initializer = RandomNormal(0, 0.02)

        self.trainers = dict()
        super().__init__(*args, **kwargs)
        self.trainer = 'gan'
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_networks(self):
        """ Add the GAN model weights """
        logger.debug("Adding networks")
        self.add_network("decoder", "a", self.decoder())
        self.add_network("decoder", "b", self.decoder())
        self.add_network("encoder", None, self.encoder())
        self.add_network("discriminator", "a", self.discriminator())
        self.add_network("discriminator", "b", self.discriminator())
        logger.debug("Added networks")

    def add_trainer(self, name, model):
        self.trainers[name] = model

    def compile_trainers(self, initialize=True):
        logger.debug("Compiling Trainers")
        learning_rate = self.config.get("learning_rate", 5e-5)
        optimizer = self.get_optimizer(lr=learning_rate, beta_1=0.5, beta_2=0.999)

        for name, model in self.trainers.items():
            combined = name.startswith("comb")
            loss_funcs = [self.loss_function(False, name, initialize)]
            if combined:
                loss_names = ["loss", "loss_a", "loss_b"]
                loss_funcs.append(self.loss_function(False, name, initialize))
                self.discriminator_a.trainable = False
                self.discriminator_b.trainable = False
            else:
                loss_names = ["loss_d"]
            model.compile(optimizer=optimizer, loss=loss_funcs)

            if initialize:
                self.state.add_session_loss_names(name, loss_names)
                self.history[name] = list()
    
    def compile_predictors(self, initialize=True):
#        super().compile_predictors(initialize)
        self.compile_trainers()

    def build_autoencoders(self):
        """ Initialize original model """
        logger.debug("Initializing model")
        inputs = [Input(shape=self.input_shape, name="face_a"), Input(shape=self.input_shape, name="face_b")]
        encoder = self.networks["encoder"].network
        decoder_a = self.networks["decoder_a"].network
        decoder_b = self.networks["decoder_b"].network
        self.discriminator_a = self.networks["discriminator_a"].network
        self.discriminator_b = self.networks["discriminator_b"].network
        decoder_a.name = 'decoder_a'
        decoder_b.name = 'decoder_b'
        self.discriminator_a.name = 'discriminator_a'
        self.discriminator_b.name = 'discriminator_b'

        generator_a = decoder_a(encoder(inputs[0]))
        generator_b = decoder_b(encoder(inputs[1]))
        predictor_a = KerasModel(inputs[0], generator_a)
        predictor_b = KerasModel(inputs[1], generator_b)
        outputs = [ self.discriminator_a(generator_a), self.discriminator_b(generator_b) ]
        self.add_predictor('a', predictor_a)
        self.add_predictor('b', predictor_b)
        self.add_trainer('a', DisModel(inputs[0], self.discriminator_a(inputs[0]), predictor_a))
        self.add_trainer('b', DisModel(inputs[1], self.discriminator_b(inputs[1]), predictor_b))
        self.add_trainer('combine', KerasModel(inputs, outputs))
        logger.debug("Initialized model")

    def encoder(self):
        """ Encoder Network """
        kwargs = dict(kernel_initializer=self.kernel_initializer)
        input_ = Input(shape=self.input_shape)
        in_conv_filters = self.input_shape[0]
        if self.input_shape[0] > 128:
            in_conv_filters = 128 + (self.input_shape[0] - 128) // 4
        dense_shape = self.input_shape[0] // 16

        o = Conv2D(64, 5, strides=(2,2), padding='same')(input_)
        o = Conv2D(128, 5, strides=(2,2), padding='same')(o)
        o = Conv2D(256, 5, strides=(2,2), padding='same')(o)
        o = Conv2D(512, 5, strides=(2,2), padding='same')(o)
        o = Flatten()(o)
        o = Dense(1024)(o)
        o = Dense(dense_shape*dense_shape*1024)(o)
        o = Reshape((dense_shape,dense_shape,1024))(o)

        o = Conv2D(2048, 3, padding='same')(o)
        y = PixelShuffler()(o)
        return KerasModel(input_, y)

        #var_x = self.blocks.conv(input_, in_conv_filters, res_block_follows=True, **kwargs)
        #tmp_x = var_x
        #res_cycles = 8 if self.config.get("lowmem", False) else 16
        #for _ in range(res_cycles):
        #    nn_x = self.blocks.res_block(var_x, 128, **kwargs)
        #    var_x = nn_x
        # consider adding scale before this layer to scale the residual chain
        #var_x = add([var_x, tmp_x])
        #var_x = self.blocks.conv(var_x, 128, **kwargs)
        #var_x = PixelShuffler()(var_x)
        #var_x = self.blocks.conv(var_x, 128, **kwargs)
        #var_x = PixelShuffler()(var_x)
        #var_x = self.blocks.conv(var_x, 128, **kwargs)
        #var_x = self.blocks.conv_sep(var_x, 256, **kwargs)
        #var_x = self.blocks.conv(var_x, 512, **kwargs)
        #if not self.config.get("lowmem", False):
        #    var_x = self.blocks.conv_sep(var_x, 1024, **kwargs)
        #
        #var_x = Dense(self.encoder_dim, **kwargs)(Flatten()(var_x))
        #var_x = Dense(dense_shape * dense_shape * 1024, **kwargs)(var_x)
        #var_x = Reshape((dense_shape, dense_shape, 1024))(var_x)
        #var_x = self.blocks.upscale(var_x, 512, **kwargs)
        #return KerasModel(input_, var_x)

    def decoder(self):
        """ Decoder Network """
        kwargs = dict(kernel_initializer=self.kernel_initializer)
        decoder_shape = self.input_shape[0] // 8
        input_ = Input(shape=(decoder_shape, decoder_shape, 512))

        upscale1 = Conv2D(1024, 3, padding='same')(input_)
        ps1 = PixelShuffler()(upscale1)
        upscale2 = Conv2D(512, 3, padding='same')(ps1)
        ps2 = PixelShuffler()(upscale2)
        upscale3 = Conv2D(256, 3, padding='same')(ps2)
        ps3 = PixelShuffler()(upscale3)
        y = Conv2D(3, 5, padding='same', activation='sigmoid')(ps3)
        return KerasModel(input_, y)

        #var_x = input_
        #var_x = self.blocks.upscale(var_x, 512, res_block_follows=True, **kwargs)
        #var_x = self.blocks.res_block(var_x, 512, **kwargs)
        #var_x = self.blocks.upscale(var_x, 256, res_block_follows=True, **kwargs)
        #var_x = self.blocks.res_block(var_x, 256, **kwargs)
        #var_x = self.blocks.upscale(var_x, self.input_shape[0], res_block_follows=True, **kwargs)
        #var_x = self.blocks.res_block(var_x, self.input_shape[0], **kwargs)
        #var_x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid", name="face_out")(var_x)
        #outputs = [var_x]
        #
        #if self.config.get("mask_type", None):
        #    var_y = input_
        #    var_y = self.blocks.upscale(var_y, 512)
        #    var_y = self.blocks.upscale(var_y, 256)
        #    var_y = self.blocks.upscale(var_y, self.input_shape[0])
        #    var_y = Conv2D(1, kernel_size=5, padding="same", activation="sigmoid", name="mask_out")(var_y)
        #    outputs.append(var_y)
        #return KerasModel(input_, outputs=outputs)

    def discriminator(self):
        kwargs = dict(kernel_initializer=self.kernel_initializer)
        input_ = Input(shape=self.input_shape)
        conv1 = Conv2D(64, 3, strides=(2,2), padding='same')(input_)
        lrelu1 = LeakyReLU()(conv1)
        conv2 = Conv2D(128, 3, strides=(2,2), padding='same')(lrelu1)
        lrelu2 = LeakyReLU()(conv2)
        conv3 = Conv2D(256, 3, strides=(2,2), padding='same')(lrelu2)
        lrelu3 = LeakyReLU()(conv3)
        conv4 = Conv2D(512, 3, strides=(2,2), padding='same')(lrelu3)
        lrelu4 = LeakyReLU()(conv4)
        conv5 = Conv2D(1024, 3, strides=(2,2), padding='same')(lrelu4)
        lrelu5 = LeakyReLU()(conv5)
        conv6 = Conv2D(1, 2, padding='same')(lrelu5)
        lrelu6 = LeakyReLU()(conv6)
        flatten = Flatten()(lrelu6)
        y = Dense(1, activation='sigmoid')(flatten)
        return KerasModel(input_, y)

class DisModel(KerasModel):
    def __init__(self, _input, _output, generator):
        super().__init__(_input, _output)
        self.generator = generator

    def train_on_batch(self, x, y, direct_input=False, sample_weight=None, class_weight=None):
        if not direct_input: x = self.generator.predict(x)
        return super().train_on_batch(x, y, sample_weight=sample_weight, class_weight=class_weight)
