#!/usr/bin/env python3
""" GAN Model """

import numpy as np
from keras.initializers import RandomNormal
from keras.layers import add, Dense, Flatten, Input, Reshape, Conv2D, concatenate
from keras.layers.advanced_activations import LeakyReLU

from keras.models import Model as KerasModel
from keras.optimizers import Adam
import keras.backend as K
from keras_vggface.vggface import VGGFace

from lib.model.layers import PixelShuffler
from lib.model.nn_blocks import upscale_ps, upscale_nn, conv_gan, conv_d_gan, res_block_gan, self_attn_block, Lambda
from lib.model.losses import adversarial_loss, reconstruction_loss, edge_loss, perceptual_loss, cyclic_loss, first_order
from ._base import ModelBase, logger

class Model(ModelBase):
    """ GAN Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        self.configfile = kwargs.get("configfile", None)
        kwargs["input_shape"] = (128, 128, 3)

        super().__init__(*args, **kwargs)
        self.trainer = 'gan'
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_networks(self):
        """ Add the GAN model weights """
        logger.debug("Adding networks")
        self.nc_G_inp = 3
        self.nc_D_inp = 6
        self.lrD = 2e-4
        self.lrG = 1e-4
        self.IMAGE_SHAPE = (128, 128, 3)
        self.output_shape = (128, 128, 3)
        self.use_self_attn = self.config.get('use_self_attn', True)
        self.norm = self.config.get('norm', 'batchnorm')
        self.model_capacity = self.config.get('model_capacity', 'standard')
        self.enc_nc_out = 256 if self.model_capacity == "lite" else 512

        self.encoder = self.build_encoder(nc_in=self.nc_G_inp,
                                          input_size=self.IMAGE_SHAPE[0],
                                          use_self_attn=self.use_self_attn,
                                          norm=self.norm,
                                          model_capacity=self.model_capacity
                                         )
        self.decoder_A = self.build_decoder(nc_in=self.enc_nc_out,
                                            input_size=8,
                                            output_size=self.IMAGE_SHAPE[0],
                                            use_self_attn=self.use_self_attn,
                                            norm=self.norm,
                                            model_capacity=self.model_capacity
                                           )
        self.decoder_B = self.build_decoder(nc_in=self.enc_nc_out,
                                            input_size=8,
                                            output_size=self.IMAGE_SHAPE[0],
                                            use_self_attn=self.use_self_attn,
                                            norm=self.norm,
                                            model_capacity=self.model_capacity
                                           )
        self.discriminator_a = self.build_discriminator(nc_in=self.nc_D_inp,
                                              input_size=self.IMAGE_SHAPE[0],
                                              use_self_attn=self.use_self_attn,
                                              norm=self.norm
                                             )
        self.discriminator_b = self.build_discriminator(nc_in=self.nc_D_inp,
                                              input_size=self.IMAGE_SHAPE[0],
                                              use_self_attn=self.use_self_attn,
                                              norm=self.norm
                                             )

        self.add_network("encoder", None, self.encoder)
        self.add_network("decoder", "a", self.decoder_A)
        self.add_network("decoder", "b", self.decoder_B)
        self.add_network("discriminator", "a", self.discriminator_a)
        self.add_network("discriminator", "b", self.discriminator_b)
        logger.debug("Added networks")

    def compile_predictors(self, initialize=True):
        for side, model in self.predictors.items():
            loss_names = ['dloss', 'ttl', 'adv', 'recon', 'edge', 'pl']
            if initialize:
                self.state.add_session_loss_names(side, loss_names)
                self.history[side] = list()

    def build_autoencoders(self):
        """ Initialize original model """
        logger.debug("Initializing model")

        encoder = self.networks['encoder'].network
        decoder_A = self.networks['decoder_a'].network
        decoder_B = self.networks['decoder_b'].network
        discriminator_a = self.networks['discriminator_a'].network
        discriminator_b = self.networks['discriminator_b'].network
        x = Input(shape=self.IMAGE_SHAPE, name='face') # dummy input tensor
        dx = Input(shape=(self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1], self.nc_D_inp)) # dummy input tensor
        self.netDA = KerasModel(dx, discriminator_a(dx))
        self.netDB = KerasModel(dx, discriminator_b(dx))
        self.netGA = KerasModel(x, decoder_A(encoder(x)))
        self.netGB = KerasModel(x, decoder_B(encoder(x)))

        # define variables
        self.distorted_A, self.fake_A, self.mask_A, \
        self.path_A, self.path_mask_A, self.path_abgr_A, self.path_bgr_A = self.define_variables(netG=self.netGA)
        self.distorted_B, self.fake_B, self.mask_B, \
        self.path_B, self.path_mask_B, self.path_abgr_B, self.path_bgr_B = self.define_variables(netG=self.netGB)
        self.real_A = Input(shape=self.IMAGE_SHAPE)
        self.real_B = Input(shape=self.IMAGE_SHAPE)
        self.mask_eyes_A = Input(shape=self.IMAGE_SHAPE)
        self.mask_eyes_B = Input(shape=self.IMAGE_SHAPE)

        # Loss function weights configuration
        loss_weights = {}
        loss_weights['w_D'] = 0.1 # Discriminator
        loss_weights['w_recon'] = 1. # L1 reconstruction loss
        loss_weights['w_edge'] = 0.1 # edge loss
        loss_weights['w_eyes'] = 30. # reconstruction and edge loss on eyes area
        loss_weights['w_pl'] = (0.01, 0.1, 0.3, 0.1) # perceptual loss (0.003, 0.03, 0.3, 0.3)

        # Init. loss config.
        loss_config = {}
        loss_config["gan_training"] = "mixup_LSGAN" # "mixup_LSGAN" or "relativistic_avg_LSGAN"
        loss_config['use_PL'] = True
        loss_config["PL_before_activ"] = False
        loss_config['use_mask_hinge_loss'] = False
        loss_config['m_mask'] = 0.
        loss_config['lr_factor'] = 1.
        loss_config['use_cyclic_loss'] = False

        # VGGFace ResNet50
        vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
        self.build_pl_model(vggface_model=vggface, before_activ=loss_config["PL_before_activ"])

        self.build_train_functions(loss_weights=loss_weights, **loss_config)

        self.add_predictor('a', self.netGA)
        self.add_predictor('b', self.netGB)
        logger.debug("Initialized model")

    def build_encoder(self, nc_in=3,
                      input_size=64,
                      use_self_attn=True,
                      norm='none',
                      model_capacity='standard'):
        coef = 2 if model_capacity == "lite" else 1
        latent_dim = 2048 if (model_capacity == "lite" and input_size > 64) else 1024
        upscale_block = upscale_nn if model_capacity == "lite" else upscale_ps
        activ_map_size = input_size
        use_norm = False if (norm == 'none') else True

        inp = Input(shape=(input_size, input_size, nc_in))
        x = Conv2D(64//coef, kernel_size=5, use_bias=False, padding="same")(inp) # use_bias should be True
        x = conv_gan(x, 128//coef)
        x = conv_gan(x, 256//coef, use_norm, norm=norm)
        x = self_attn_block(x, 256//coef) if use_self_attn else x
        x = conv_gan(x, 512//coef, use_norm, norm=norm)
        x = self_attn_block(x, 512//coef) if use_self_attn else x
        x = conv_gan(x, 1024//(coef**2), use_norm, norm=norm)

        activ_map_size = activ_map_size//16
        while (activ_map_size > 4):
            x = conv_gan(x, 1024//(coef**2), use_norm, norm=norm)
            activ_map_size = activ_map_size//2

        x = Dense(latent_dim)(Flatten()(x))
        x = Dense(4*4*1024//(coef**2))(x)
        x = Reshape((4, 4, 1024//(coef**2)))(x)
        out = upscale_block(x, 512//coef, use_norm, norm=norm)
        return KerasModel(inputs=inp, outputs=out)

    def build_decoder(self, nc_in=512,
                      input_size=8,
                      output_size=64,
                      use_self_attn=True,
                      norm='none',
                      model_capacity='standard'):
        coef = 2 if model_capacity == "lite" else 1
        upscale_block = upscale_nn if model_capacity == "lite" else upscale_ps
        activ_map_size = input_size
        use_norm = False if (norm == 'none') else True

        inp = Input(shape=(input_size, input_size, nc_in))
        x = inp
        x = upscale_block(x, 256//coef, use_norm, norm=norm)
        x = upscale_block(x, 128//coef, use_norm, norm=norm)
        x = self_attn_block(x, 128//coef) if use_self_attn else x
        x = upscale_block(x, 64//coef, use_norm, norm=norm)
        x = res_block_gan(x, 64//coef, norm=norm)
        x = self_attn_block(x, 64//coef) if use_self_attn else conv_gan(x, 64//coef, strides=1)

        outputs = []
        activ_map_size = activ_map_size * 8
        while (activ_map_size < output_size):
            outputs.append(Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x))
            x = upscale_block(x, 64//coef, use_norm, norm=norm)
            x = conv_gan(x, 64//coef, strides=1)
            activ_map_size *= 2

        alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
        bgr = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
        out = concatenate([alpha, bgr])
        outputs.append(out)
        return KerasModel(inp, outputs)

    def build_discriminator(self, nc_in,
                            input_size=64,
                            use_self_attn=True,
                            norm='none'):
        activ_map_size = input_size
        use_norm = False if (norm == 'none') else True

        inp = Input(shape=(input_size, input_size, nc_in))
        x = conv_d_gan(inp, 64, False)
        x = conv_d_gan(x, 128, use_norm, norm=norm)
        x = conv_d_gan(x, 256, use_norm, norm=norm)
        x = self_attn_block(x, 256) if use_self_attn else x

        activ_map_size = activ_map_size//8
        while (activ_map_size > 8):
            x = conv_d_gan(x, 256, use_norm, norm=norm)
            x = self_attn_block(x, 256) if use_self_attn else x
            activ_map_size = activ_map_size//2

        out = Conv2D(1, kernel_size=4, use_bias=False, padding="same")(x) # use_bias should be True
        return KerasModel(inputs=[inp], outputs=out)

    def define_variables(self, netG):
        distorted_input = netG.inputs[0]
        fake_output = netG.outputs[-1]
        alpha = Lambda(lambda x: x[:,:,:, :1])(fake_output)
        bgr = Lambda(lambda x: x[:,:,:, 1:])(fake_output)

        masked_fake_output = alpha * bgr + (1-alpha) * distorted_input

        fn_generate = K.function([distorted_input], [masked_fake_output])
        fn_mask = K.function([distorted_input], [concatenate([alpha, alpha, alpha])])
        fn_abgr = K.function([distorted_input], [concatenate([alpha, bgr])])
        fn_bgr = K.function([distorted_input], [bgr])
        return distorted_input, fake_output, alpha, fn_generate, fn_mask, fn_abgr, fn_bgr

    def build_train_functions(self, loss_weights=None, **loss_config):
        assert loss_weights is not None, "loss weights are not provided."
        # Adversarial loss
        loss_DA, loss_adv_GA = adversarial_loss(self.netDA, self.real_A, self.fake_A,
                                                self.distorted_A,
                                                loss_config["gan_training"],
                                                **loss_weights)
        loss_DB, loss_adv_GB = adversarial_loss(self.netDB, self.real_B, self.fake_B,
                                                self.distorted_B,
                                                loss_config["gan_training"],
                                                **loss_weights)

        # Reconstruction loss
        loss_recon_GA = reconstruction_loss(self.real_A, self.fake_A,
                                            self.mask_eyes_A, self.netGA.outputs,
                                            **loss_weights)
        loss_recon_GB = reconstruction_loss(self.real_B, self.fake_B,
                                            self.mask_eyes_B, self.netGB.outputs,
                                            **loss_weights)

        # Edge loss
        loss_edge_GA = edge_loss(self.real_A, self.fake_A, self.mask_eyes_A, **loss_weights)
        loss_edge_GB = edge_loss(self.real_B, self.fake_B, self.mask_eyes_B, **loss_weights)

        if loss_config['use_PL']:
            loss_pl_GA = perceptual_loss(self.real_A, self.fake_A, self.distorted_A,
                                         self.mask_eyes_A, self.vggface_feats, **loss_weights)
            loss_pl_GB = perceptual_loss(self.real_B, self.fake_B, self.distorted_B,
                                         self.mask_eyes_B, self.vggface_feats, **loss_weights)
        else:
            loss_pl_GA = loss_pl_GB = K.zeros(1)

        loss_GA = loss_adv_GA + loss_recon_GA + loss_edge_GA + loss_pl_GA
        loss_GB = loss_adv_GB + loss_recon_GB + loss_edge_GB + loss_pl_GB

        # The following losses are rather trivial, thus their wegihts are fixed.
        # Cycle consistency loss
        if loss_config['use_cyclic_loss']:
            loss_GA += 10 * cyclic_loss(self.netGA, self.netGB, self.real_A)
            loss_GB += 10 * cyclic_loss(self.netGB, self.netGA, self.real_B)

        # Alpha mask loss
        if not loss_config['use_mask_hinge_loss']:
            loss_GA += 1e-2 * K.mean(K.abs(self.mask_A))
            loss_GB += 1e-2 * K.mean(K.abs(self.mask_B))
        else:
            loss_GA += 0.1 * K.mean(K.maximum(0., loss_config['m_mask'] - self.mask_A))
            loss_GB += 0.1 * K.mean(K.maximum(0., loss_config['m_mask'] - self.mask_B))

        # Alpha mask total variation loss
        loss_GA += 0.1 * K.mean(first_order(self.mask_A, axis=1))
        loss_GA += 0.1 * K.mean(first_order(self.mask_A, axis=2))
        loss_GB += 0.1 * K.mean(first_order(self.mask_B, axis=1))
        loss_GB += 0.1 * K.mean(first_order(self.mask_B, axis=2))

        # L2 weight decay
        # https://github.com/keras-team/keras/issues/2662
        for loss_tensor in self.netGA.losses:
            loss_GA += loss_tensor
        for loss_tensor in self.netGB.losses:
            loss_GB += loss_tensor
        for loss_tensor in self.netDA.losses:
            loss_DA += loss_tensor
        for loss_tensor in self.netDB.losses:
            loss_DB += loss_tensor

        weightsDA = self.netDA.trainable_weights
        weightsGA = self.netGA.trainable_weights
        weightsDB = self.netDB.trainable_weights
        weightsGB = self.netGB.trainable_weights

        # Define training functions
        # Adam(...).get_updates(...)
        training_updates = Adam(lr=self.lrD*loss_config['lr_factor'], beta_1=0.5).get_updates(weightsDA,[],loss_DA)
        self.netDA_train = K.function([self.distorted_A, self.real_A],[loss_DA], training_updates)
        training_updates = Adam(lr=self.lrG*loss_config['lr_factor'], beta_1=0.5).get_updates(weightsGA,[], loss_GA)
        self.netGA_train = K.function([self.distorted_A, self.real_A, self.mask_eyes_A],
                                      [loss_GA, loss_adv_GA, loss_recon_GA, loss_edge_GA, loss_pl_GA],
                                      training_updates)

        training_updates = Adam(lr=self.lrD*loss_config['lr_factor'], beta_1=0.5).get_updates(weightsDB,[],loss_DB)
        self.netDB_train = K.function([self.distorted_B, self.real_B],[loss_DB], training_updates)
        training_updates = Adam(lr=self.lrG*loss_config['lr_factor'], beta_1=0.5).get_updates(weightsGB,[], loss_GB)
        self.netGB_train = K.function([self.distorted_B, self.real_B, self.mask_eyes_B],
                                      [loss_GB, loss_adv_GB, loss_recon_GB, loss_edge_GB, loss_pl_GB],
                                      training_updates)

    def build_pl_model(self, vggface_model, before_activ=False):
        # Define Perceptual Loss Model
        vggface_model.trainable = False
        if before_activ == False:
            out_size112 = vggface_model.layers[1].output
            out_size55 = vggface_model.layers[36].output
            out_size28 = vggface_model.layers[78].output
            out_size7 = vggface_model.layers[-2].output
        else:
            out_size112 = vggface_model.layers[15].output # misnamed: the output size is 55
            out_size55 = vggface_model.layers[35].output
            out_size28 = vggface_model.layers[77].output
            out_size7 = vggface_model.layers[-3].output
        self.vggface_feats = KerasModel(vggface_model.input, [out_size112, out_size55, out_size28, out_size7])
        self.vggface_feats.trainable = False

    def train_one_batch_G(self, data_A, data_B):
        if len(data_A) == 4 and len(data_B) == 4:
            _, warped_A, target_A, bm_eyes_A = data_A
            _, warped_B, target_B, bm_eyes_B = data_B
        elif len(data_A) == 3 and len(data_B) == 3:
            warped_A, target_A, bm_eyes_A = data_A
            warped_B, target_B, bm_eyes_B = data_B
        elif len(data_A) == 2 and len(data_B) == 2:
            warped_A, target_A = data_A
            warped_B, target_B = data_B
            bm_eyes_A = np.zeros_like(warped_A)
            bm_eyes_B = np.zeros_like(warped_B)
        else:
            raise ValueError("Something's wrong with the input data generator.")
        errGA = self.netGA_train([warped_A, target_A, bm_eyes_A])
        errGB = self.netGB_train([warped_B, target_B, bm_eyes_B])
        return errGA, errGB

    def train_one_batch_D(self, data_A, data_B):
        if len(data_A) == 4 and len(data_B) == 4:
            _, warped_A, target_A, _ = data_A
            _, warped_B, target_B, _ = data_B
        elif len(data_A) == 3 and len(data_B) == 3:
            warped_A, target_A, _ = data_A
            warped_B, target_B, _ = data_B
        elif len(data_A) == 2 and len(data_B) == 2:
            warped_A, target_A = data_A
            warped_B, target_B = data_B
        else:
            raise ValueError("Something's wrong with the input data generator.")
        errDA = self.netDA_train([warped_A, target_A])
        errDB = self.netDB_train([warped_B, target_B])
        return errDA, errDB

    def transform_A2B(self, img):
        return self.path_abgr_B([[img]])

    def transform_B2A(self, img):
        return self.path_abgr_A([[img]])
