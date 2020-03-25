'''
Sefika Efeoglu
'''
import tensorflow as tf
import numpy as np
import glob, random, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class VariationalAutoencoderConfigBase(object):
    '''Configuration General Class'''
    
    def __init__(self, config=1):
        
        if config == 2:
            self.encoder_filter = [4, 8, 16, 32]
            self.decoder_filter = [32, 16, 4, 3]
        elif config == 3:
            self.encoder_filter = [16, 32, 64, 128]
            self.decoder_filter = [64, 32, 16, 3]
        elif config == 4:
            self.encoder_filter = [64, 128, 256, 512]
            self.decoder_filter = [256, 128, 64, 3]
        elif config == 1:
            self.encoder_filter = [32, 64, 128, 256]
            self.decoder_filter = [128, 64, 32, 3]

        self.image = tf.placeholder(tf.float32, [None, 96, 96, 3], name='image')
        self.resized_image = tf.image.resize_images(self.image, [64, 64])
        tf.summary.image('resized_image', self.resized_image, 20)

        self.z_mu, self.z_logvar = self.encoder(self.resized_image,self.encoder_filter)
        self.z = self.sample_z(self.z_mu, self.z_logvar)
        self.reconstructions = self.decoder(self.z, self.decoder_filter)
        tf.summary.image('reconstructions', self.reconstructions, 20)

        self.merged = tf.summary.merge_all()

        self.loss = self.compute_loss()

    def sample_z(self, mu, logvar):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(logvar / 2) * eps


    def compute_loss(self):
        logits_flat = tf.layers.flatten(self.reconstructions)
        labels_flat = tf.layers.flatten(self.resized_image)
        reconstruction_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis = 1)
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.z_logvar) + self.z_mu**2 - 1. - self.z_logvar, 1)
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        return vae_loss

    def encoder(self, x, filter_list):
        x = tf.layers.conv2d(x, filters=filter_list[0], kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=filter_list[1], kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=filter_list[2], kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=filter_list[3], kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)

        x = tf.layers.flatten(x)
        z_mu = tf.layers.dense(x, units=32, name='z_mu')
        z_logvar = tf.layers.dense(x, units=32, name='z_logvar')
        return z_mu, z_logvar

    def decoder(self, z, filter_list):
        x = tf.layers.dense(z, 1024, activation=None)
        x = tf.reshape(x, [-1, 1, 1, 1024])
        x = tf.layers.conv2d_transpose(x, filters=filter_list[0], kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, filters=filter_list[1], kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, filters=filter_list[2], kernel_size=6, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, filters=filter_list[3], kernel_size=6, strides=2, padding='valid', activation=tf.nn.sigmoid)
        return x
