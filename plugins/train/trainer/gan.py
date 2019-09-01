#!/usr/bin/env python3


""" GAN Trainer Class for Faceswap """

import logging
import os
import time

import cv2
import numpy as np

from tensorflow import keras as tf_keras

from lib.alignments import Alignments
from lib.faces_detect import DetectedFace
from lib.training_data import TrainingDataGenerator, stack_images
from lib.utils import get_folder, get_image_paths

from ._base import TrainerBase

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Trainer(TrainerBase):
    """ GAN Trainer """

    def __init__(self, model, images, batch_size):
        logger.debug("Initializing %s: (model: '%s', batch_size: %s)",
                     self.__class__.__name__, model, batch_size)
        super().__init__(model, images, batch_size)
        self.sides = sorted(key for key in self.images.keys())
        self.training_sides = sorted(key for key in self.model.trainers.keys())

        self.batchers = {side: Batcher(side,
                                       images if side.startswith('comb') else images[side],
                                       self.model,
                                       self.use_mask,
                                       batch_size)
                         for side in self.training_sides}

        self.tensorboard = self.set_tensorboard()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def timestamp(self):
        """ Standardised timestamp for loss reporting """
        return time.strftime("%H:%M:%S")

    @property
    def landmarks_required(self):
        """ Return True if Landmarks are required """
        opts = self.model.training_opts
        retval = bool(opts.get("mask_type", None) or opts["warp_to_landmarks"])
        logger.debug(retval)
        return retval

    @property
    def use_mask(self):
        """ Return True if a mask is requested """
        retval = bool(self.model.training_opts.get("mask_type", None))
        logger.debug(retval)
        return retval

    def set_tensorboard(self):
        """ Set up tensorboard callback """
        if self.model.training_opts["no_logs"]:
            logger.verbose("TensorBoard logging disabled")
            return None
        if self.pingpong.active:
            # Currently TensorBoard uses the tf.session, meaning that VRAM does not
            # get cleared when model switching
            # TODO find a fix for this
            logger.warning("Currently TensorBoard logging is not supported for Ping-Pong "
                           "training. Session stats and graphing will not be available for this "
                           "training session.")
            return None

        logger.debug("Enabling TensorBoard Logging")
        tensorboard = dict()
        for side in self.sides:
            logger.debug("Setting up TensorBoard Logging. Side: %s", side)
            log_dir = os.path.join(str(self.model.model_dir),
                                   "{}_logs".format(self.model.name),
                                   side,
                                   "session_{}".format(self.model.state.session_id))
            tbs = tf_keras.callbacks.TensorBoard(log_dir=log_dir,
                                                 histogram_freq=0,  # Must be 0 or hangs
                                                 batch_size=self.batch_size,
                                                 write_graph=True,
                                                 write_grads=True)
            tbs.set_model(self.model.predictors[side])
            tensorboard[side] = tbs
        logger.info("Enabled TensorBoard Logging")
        return tensorboard

    def print_loss(self, loss):
        """ Override for specific model loss formatting """
        logger.trace(loss)
        output = list()
        for side in sorted(list(loss.keys())):
            display = ", ".join(["{}_{}: {:.5f}".format(self.model.state.loss_names[side][idx],
                                                        side.capitalize(),
                                                        this_loss)
                                 for idx, this_loss in enumerate(loss[side])])
            output.append(display)
        output = ", ".join(output)
        print("[{}] [#{:05d}] {}".format(self.timestamp, self.model.iterations, output), end='\r')

    def train_one_step(self, viewer, timelapse_kwargs):
        """ Train a batch """
        logger.trace("Training one step: (iteration: %s)", self.model.iterations)
        do_preview = False if viewer is None else True
        loss = dict()
        for side, batcher in self.batchers.items():
            if self.pingpong.active and side != self.pingpong.side:
                continue
            loss[side] = batcher.train_one_batch(do_preview)

        self.model.state.increment_iterations()

        for side, side_loss in loss.items():
            self.store_history(side, side_loss)
            self.log_tensorboard(side, side_loss)

        if not self.pingpong.active:
            self.print_loss(loss)
        else:
            for key, val in loss.items():
                self.pingpong.loss[key] = val
            self.print_loss(self.pingpong.loss)

    def store_history(self, side, loss):
        """ Store the history of this step """
        logger.trace("Updating loss history: '%s'", side)
        self.model.history[side].append(loss[0])  # Either only loss or total loss
        logger.trace("Updated loss history: '%s'", side)

    def log_tensorboard(self, side, loss):
        """ Log loss to TensorBoard log """
        if not self.tensorboard:
            return
        logger.trace("Updating TensorBoard log: '%s'", side)
        logs = {log[0]: log[1]
                for log in zip(self.model.state.loss_names[side], loss)}
        self.tensorboard[side].on_batch_end(self.model.state.iterations, logs)
        logger.trace("Updated TensorBoard log: '%s'", side)

    def clear_tensorboard(self):
        """ Indicate training end to Tensorboard """
        if not self.tensorboard:
            return
        for side, tensorboard in self.tensorboard.items():
            logger.debug("Ending Tensorboard. Side: '%s'", side)
            tensorboard.on_train_end(None)


class Batcher():
    """ Batch images from a single side """
    def __init__(self, side, images, model, use_mask, batch_size):
        logger.debug("Initializing %s: side: '%s', num_images: %s, batch_size: %s)",
                     self.__class__.__name__, side, len(images), batch_size)
        self.model = model
        self.use_mask = use_mask
        self.side = side
        self.target = None
        self.mask = None
        self.valid = np.ones((batch_size, 1))
        self.fake = np.zeros((batch_size, 1))
        if isinstance(images, dict):
            self.is_generator = True
            self.feed = [self.load_generator().minibatch_ab(images[side], batch_size, side) for side in images.keys()]
        else:
            self.is_generator = False
            self.feed = [self.load_generator().minibatch_ab(images, batch_size, self.side)]

    def load_generator(self):
        """ Pass arguments to TrainingDataGenerator and return object """
        logger.debug("Loading generator: %s", self.side)
        input_size = self.model.input_shape[0]
        output_size = self.model.output_shape[0]
        logger.debug("input_size: %s, output_size: %s", input_size, output_size)
        generator = TrainingDataGenerator(input_size, output_size, self.model.training_opts)
        return generator

    def train_one_batch(self, do_preview):
        """ Train a batch """
        logger.trace("Training one step: (side: %s)", self.side)
        x, y = self.get_next(do_preview)
        if self.is_generator:
            loss = self.model.trainers[self.side].train_on_batch(x, [self.valid, self.valid])
        else:
            loss = self.model.trainers[self.side].train_on_batch(y, self.valid)
            loss += self.model.trainers[self.side].train_on_batch(x, self.fake)
            loss *= 0.5
        if not isinstance(loss, list): loss = [loss]
        return loss

    def get_next(self, do_preview):
        """ Return the next batch from the generator
            Items should come out as: (warped, target [, mask]) """
        xs = []
        ys = []
        for feed in self.feed:
            _, x, y = next(feed)
            xs.append(x)
            ys.append(y)
        return xs, ys

    def compile_mask(self, batch):
        """ Compile the mask into training data """
        logger.trace("Compiling Mask: (side: '%s')", self.side)
        mask = batch[-1]
        retval = list()
        for idx in range(len(batch) - 1):
            image = batch[idx]
            retval.append([image, mask])
        return retval
