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

        self.batcher = Batcher(images, self.model, self.use_mask, batch_size)

        self.tensorboard = self.set_tensorboard()
        self.samples = Samples(self.model,
                               self.use_mask,
                               self.model.training_opts["coverage_ratio"],
                               self.model.training_opts["preview_scaling"])
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
        loss = self.batcher.train_one_batch(do_preview)
        if do_preview:
            self.samples.images = self.batcher.compile_sample(self.batch_size)

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

        if do_preview:
            samples = self.samples.show_sample()
            if samples is not None:
                viewer(samples, "Training - 'S': Save Now. 'ENTER': Save and Quit")

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
    def __init__(self, images, model, use_mask, batch_size):
        logger.debug("Initializing %s: num_images: %s, batch_size: %s)",
                     self.__class__.__name__, len(images), batch_size)
        self.model = model
        self.use_mask = use_mask
        self.target = None
        self.mask = None
        self.valid = np.ones((batch_size, 1))
        self.fake = np.zeros((batch_size, 1))
        self.feed = dict()
        for side in images.keys():
            self.feed[side] = self.load_generator().minibatch_ab(images[side], batch_size, side)

    def load_generator(self):
        """ Pass arguments to TrainingDataGenerator and return object """
        logger.debug("Loading generator")
        input_size = self.model.input_shape[0]
        output_size = self.model.output_shape[0]
        logger.debug("input_size: %s, output_size: %s", input_size, output_size)
        generator = TrainingDataGenerator(input_size, output_size, self.model.training_opts)
        return generator

    def train_one_batch(self, do_preview):
        """ Train a batch """
        logger.trace("Training one step")
        data_a, data_b = self.get_next(do_preview)
        errDA, errDB = self.model.train_one_batch_D(data_A=data_a, data_B=data_b)

        data_a, data_b = self.get_next(do_preview)
        errGA, errGB = self.model.train_one_batch_G(data_A=data_a, data_B=data_b)

        loss = { 'a': [ errDA[0] ], 'b': [ errDB[0] ] }
        for i, loss_name in enumerate(['ttl', 'adv', 'recon', 'edge', 'pl']):
            la = errGA[i][0] if loss_name in ['ttl', 'pl'] else errGA[i]
            lb = errGB[i][0] if loss_name in ['ttl', 'pl'] else errGB[i]
            loss['a'].append(la)
            loss['b'].append(lb)
        return loss

    def get_next(self, do_preview):
        """ Return the next batch from the generator
            Items should come out as: (warped, target [, mask]) """
        samples_a, x_a, y_a = next(self.feed['a'])
        samples_b, x_b, y_b = next(self.feed['b'])
        self.samples = [samples_a, samples_b] if do_preview else None
        self.target = [x_a, x_b] if do_preview else None
        return [x_a, y_a], [x_b, y_b]

    def compile_mask(self, batch):
        """ Compile the mask into training data """
        logger.trace("Compiling Mask")
        mask = batch[-1]
        retval = list()
        for idx in range(len(batch) - 1):
            image = batch[idx]
            retval.append([image, mask])
        return retval

    def compile_sample(self, batch_size, samples=None, images=None):
        """ Training samples to display in the viewer """
        num_images = self.model.training_opts.get("preview_images", 14)
        num_images = min(batch_size, num_images)
        logger.debug("Compiling samples: (samples: %s)", num_images)
        ret = dict()
        for i, side in enumerate(['a', 'b']):
            images = images if images is not None else self.target[i]
            sample = [samples[0:num_images]] if samples is not None else [self.samples[i][0:num_images]]
            if self.use_mask:
                retval = [tgt[0:num_images] for tgt in images]
            else:
                retval = [images[0:num_images]]
            retval = sample + retval
            ret[side] = retval
        return ret

class Samples():
    """ Display samples for preview and timelapse """
    def __init__(self, model, use_mask, coverage_ratio, scaling=1.0):
        logger.debug("Initializing %s: model: '%s', use_mask: %s, coverage_ratio: %s)",
                     self.__class__.__name__, model, use_mask, coverage_ratio)
        self.model = model
        self.use_mask = use_mask
        self.images = dict()
        self.coverage_ratio = coverage_ratio
        self.scaling = scaling
        logger.debug("Initialized %s", self.__class__.__name__)

    def show_sample(self):
        """ Display preview data """
        if len(self.images) != 2:
            logger.debug("Ping Pong training - Only one side trained. Aborting preview")
            return None
        logger.debug("Showing sample")
        feeds = dict()
        figures = dict()
        headers = dict()
        for side, samples in self.images.items():
            faces = samples[1]
            if self.model.input_shape[0] / faces.shape[1] != 1.0:
                feeds[side] = self.resize_sample(side, faces, self.model.input_shape[0])
                feeds[side] = feeds[side].reshape((-1, ) + self.model.input_shape)
            else:
                feeds[side] = faces
            if self.use_mask:
                mask = samples[-1]
                feeds[side] = [feeds[side], mask]

        preds = self.get_predictions(feeds["a"], feeds["b"])

        for side, samples in self.images.items():
            other_side = "a" if side == "b" else "b"
            predictions = [preds["{}_{}".format(side, side)],
                           preds["{}_{}".format(other_side, side)]]
            display = self.to_full_frame(side, samples, predictions)
            headers[side] = self.get_headers(side, other_side, display[0].shape[1])
            figures[side] = np.stack([display[0], display[1], display[2], ], axis=1)
            if self.images[side][0].shape[0] % 2 == 1:
                figures[side] = np.concatenate([figures[side],
                                                np.expand_dims(figures[side][0], 0)])

        width = 4
        side_cols = width // 2
        if side_cols != 1:
            headers = self.duplicate_headers(headers, side_cols)

        header = np.concatenate([headers["a"], headers["b"]], axis=1)
        figure = np.concatenate([figures["a"], figures["b"]], axis=0)
        height = int(figure.shape[0] / width)
        figure = figure.reshape((width, height) + figure.shape[1:])
        figure = stack_images(figure)
        figure = np.vstack((header, figure))

        logger.debug("Compiled sample")
        return np.clip(figure * 127.0 + 127.0, 0, 255).astype('uint8')

    @staticmethod
    def resize_sample(side, sample, target_size):
        """ Resize samples where predictor expects different shape from processed image """
        scale = target_size / sample.shape[1]
        if scale == 1.0:
            return sample
        logger.debug("Resizing sample: (side: '%s', sample.shape: %s, target_size: %s, scale: %s)",
                     side, sample.shape, target_size, scale)
        interpn = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA  # pylint: disable=no-member
        retval = np.array([cv2.resize(img,  # pylint: disable=no-member
                                      (target_size, target_size),
                                      interpn)
                           for img in sample])
        logger.debug("Resized sample: (side: '%s' shape: %s)", side, retval.shape)
        return retval

    def get_predictions(self, feed_a, feed_b):
        """ Return the sample predictions from the model """
        logger.debug("Getting Predictions")
        preds = dict()
        preds["a_a"] = self.model.predictors["a"].predict(feed_a)
        preds["b_a"] = self.model.predictors["b"].predict(feed_a)
        preds["a_b"] = self.model.predictors["a"].predict(feed_b)
        preds["b_b"] = self.model.predictors["b"].predict(feed_b)

        # Get the returned image from predictors that emit multiple items
        if not isinstance(preds["a_a"], np.ndarray):
            for key, val in preds.items():
                preds[key] = val[0]
        logger.debug("Returning predictions: %s", {key: val.shape for key, val in preds.items()})
        return preds

    def to_full_frame(self, side, samples, predictions):
        """ Patch the images into the full frame """
        logger.debug("side: '%s', number of sample arrays: %s, prediction.shapes: %s)",
                     side, len(samples), [pred.shape for pred in predictions])
        full, faces = samples[:2]
        images = [faces] + predictions
        full_size = full.shape[1]
        target_size = int(full_size * self.coverage_ratio)
        if target_size != full_size:
            frame = self.frame_overlay(full, target_size, (0, 0, 255))

        if self.use_mask:
            images = self.compile_masked(images, samples[-1])
        images = [self.resize_sample(side, image, target_size) for image in images]
        if target_size != full_size:
            images = [self.overlay_foreground(frame, image) for image in images]
        if self.scaling != 1.0:
            new_size = int(full_size * self.scaling)
            images = [self.resize_sample(side, image, new_size) for image in images]
        return images

    @staticmethod
    def frame_overlay(images, target_size, color):
        """ Add roi frame to a backfround image """
        logger.debug("full_size: %s, target_size: %s, color: %s",
                     images.shape[1], target_size, color)
        new_images = list()
        full_size = images.shape[1]
        padding = (full_size - target_size) // 2
        length = target_size // 4
        t_l, b_r = (padding, full_size - padding)
        for img in images:
            cv2.rectangle(img,  # pylint: disable=no-member
                          (t_l, t_l),
                          (t_l + length, t_l + length),
                          color,
                          3)
            cv2.rectangle(img,  # pylint: disable=no-member
                          (b_r, t_l),
                          (b_r - length, t_l + length),
                          color,
                          3)
            cv2.rectangle(img,  # pylint: disable=no-member
                          (b_r, b_r),
                          (b_r - length,
                           b_r - length),
                          color,
                          3)
            cv2.rectangle(img,  # pylint: disable=no-member
                          (t_l, b_r),
                          (t_l + length, b_r - length),
                          color,
                          3)
            new_images.append(img)
        retval = np.array(new_images)
        logger.debug("Overlayed background. Shape: %s", retval.shape)
        return retval

    @staticmethod
    def compile_masked(faces, masks):
        """ Add the mask to the faces for masked preview """
        retval = list()
        masks3 = np.tile(1 - np.rint(masks), 3)
        for mask in masks3:
            mask[np.where((mask == [1., 1., 1.]).all(axis=2))] = [0., 0., 1.]
        for previews in faces:
            images = np.array([cv2.addWeighted(img, 1.0,  # pylint: disable=no-member
                                               masks3[idx], 0.3,
                                               0)
                               for idx, img in enumerate(previews)])
            retval.append(images)
        logger.debug("masked shapes: %s", [faces.shape for faces in retval])
        return retval

    @staticmethod
    def overlay_foreground(backgrounds, foregrounds):
        """ Overlay the training images into the center of the background """
        offset = (backgrounds.shape[1] - foregrounds.shape[1]) // 2
        new_images = list()
        for idx, img in enumerate(backgrounds):
            img[offset:offset + foregrounds[idx].shape[0],
                offset:offset + foregrounds[idx].shape[1]] = foregrounds[idx]
            new_images.append(img)
        retval = np.array(new_images)
        logger.debug("Overlayed foreground. Shape: %s", retval.shape)
        return retval

    def get_headers(self, side, other_side, width):
        """ Set headers for images """
        logger.debug("side: '%s', other_side: '%s', width: %s",
                     side, other_side, width)
        side = side.upper()
        other_side = other_side.upper()
        height = int(64 * self.scaling)
        total_width = width * 3
        logger.debug("height: %s, total_width: %s", height, total_width)
        font = cv2.FONT_HERSHEY_SIMPLEX  # pylint: disable=no-member
        texts = ["Target {}".format(side),
                 "{} > {}".format(side, side),
                 "{} > {}".format(side, other_side)]
        text_sizes = [cv2.getTextSize(texts[idx],  # pylint: disable=no-member
                                      font,
                                      self.scaling,
                                      1)[0]
                      for idx in range(len(texts))]
        text_y = int((height + text_sizes[0][1]) / 2)
        text_x = [int((width - text_sizes[idx][0]) / 2) + width * idx
                  for idx in range(len(texts))]
        logger.debug("texts: %s, text_sizes: %s, text_x: %s, text_y: %s",
                     texts, text_sizes, text_x, text_y)
        header_box = np.ones((height, total_width, 3), np.float32)
        for idx, text in enumerate(texts):
            cv2.putText(header_box,  # pylint: disable=no-member
                        text,
                        (text_x[idx], text_y),
                        font,
                        self.scaling,
                        (0, 0, 0),
                        1,
                        lineType=cv2.LINE_AA)  # pylint: disable=no-member
        logger.debug("header_box.shape: %s", header_box.shape)
        return header_box

    @staticmethod
    def duplicate_headers(headers, columns):
        """ Duplicate headers for the number of columns displayed """
        for side, header in headers.items():
            duped = tuple([header for _ in range(columns)])
            headers[side] = np.concatenate(duped, axis=1)
            logger.debug("side: %s header.shape: %s", side, header.shape)
        return headers


class Timelapse():
    """ Create the timelapse """
    def __init__(self, model, use_mask, coverage_ratio, batchers):
        logger.debug("Initializing %s: model: %s, use_mask: %s, coverage_ratio: %s, "
                     "batchers: '%s')", self.__class__.__name__, model, use_mask,
                     coverage_ratio, batchers)
        self.samples = Samples(model, use_mask, coverage_ratio)
        self.model = model
        self.batchers = batchers
        self.output_file = None
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_sample(self, side, timelapse_kwargs):
        """ Perform timelapse """
        logger.debug("Getting timelapse samples: '%s'", side)
        if not self.output_file:
            self.setup(**timelapse_kwargs)
        self.samples.images[side] = self.batchers[side].compile_timelapse_sample()
        logger.debug("Got timelapse samples: '%s' - %s", side, len(self.samples.images[side]))

    def setup(self, input_a=None, input_b=None, output=None):
        """ Set the timelapse output folder """
        logger.debug("Setting up timelapse")
        if output is None:
            output = str(get_folder(os.path.join(str(self.model.model_dir),
                                                 "{}_timelapse".format(self.model.name))))
        self.output_file = str(output)
        logger.debug("Timelapse output set to '%s'", self.output_file)

        images = {"a": get_image_paths(input_a), "b": get_image_paths(input_b)}
        batchsize = min(len(images["a"]),
                        len(images["b"]),
                        self.model.training_opts.get("preview_images", 14))
        for side, image_files in images.items():
            self.batchers[side].set_timelapse_feed(image_files, batchsize)
        logger.debug("Set up timelapse")

    def output_timelapse(self):
        """ Set the timelapse dictionary """
        logger.debug("Ouputting timelapse")
        image = self.samples.show_sample()
        if image is None:
            return
        filename = os.path.join(self.output_file, str(int(time.time())) + ".jpg")

        cv2.imwrite(filename, image)  # pylint: disable=no-member
        logger.debug("Created timelapse: '%s'", filename)


class PingPong():
    """ Side switcher for pingpong training """
    def __init__(self, model, sides):
        logger.debug("Initializing %s: (model: '%s')", self.__class__.__name__, model)
        self.active = model.training_opts.get("pingpong", False)
        self.model = model
        self.sides = sides
        self.side = sorted(sides)[0]
        self.loss = {side: dict() for side in sides}
        logger.debug("Initialized %s", self.__class__.__name__)

    def switch(self):
        """ Switch pingpong side """
        if not self.active:
            return
        retval = [side for side in self.sides if side != self.side][0]
        logger.info("Switching training to side %s", retval.title())
        self.side = retval
        self.reload_model()

    def reload_model(self):
        """ Load the model for just the current side """
        logger.verbose("Ping-Pong re-loading model")
        self.model.reset_pingpong()


class Landmarks():
    """ Set Landmarks for training into the model's training options"""
    def __init__(self, training_opts):
        logger.debug("Initializing %s: (training_opts: '%s')",
                     self.__class__.__name__, training_opts)
        self.size = training_opts.get("training_size", 256)
        self.paths = training_opts["alignments"]
        self.landmarks = self.get_alignments()
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_alignments(self):
        """ Obtain the landmarks for each faceset """
        landmarks = dict()
        for side, fullpath in self.paths.items():
            path, filename = os.path.split(fullpath)
            filename, extension = os.path.splitext(filename)
            serializer = extension[1:]
            alignments = Alignments(
                path,
                filename=filename,
                serializer=serializer)
            landmarks[side] = self.transform_landmarks(alignments)
        return landmarks

    def transform_landmarks(self, alignments):
        """ For each face transform landmarks and return """
        landmarks = dict()
        for _, faces, _, _ in alignments.yield_faces():
            for face in faces:
                detected_face = DetectedFace()
                detected_face.from_alignment(face)
                detected_face.load_aligned(None, size=self.size, align_eyes=False)
                landmarks[detected_face.hash] = detected_face.aligned_landmarks
        return landmarks
