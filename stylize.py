# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import vgg

import tensorflow as tf
import numpy as np
import random

from sys import stderr

from PIL import Image

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

try:
    reduce
except NameError:
    from functools import reduce


def stylize(network, initial, initial_noiseblend, content, styles, preserve_colors, iterations,
        content_weight, content_weight_blend, style_weight, style_layer_weight_exp, style_blend_weights, tv_weight,
        learning_rate, beta1, beta2, epsilon, pooling,
        print_iterations=None, checkpoint_iterations=None):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)

    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight
        layer_weight *= style_layer_weight_exp

    # normalize style layer weights
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum

    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    # compute style features in feedforward mode
    maps = [{} for _ in styles]
    for i in range(len(styles)):
        g = tf.Graph()
        with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
            image = tf.placeholder('float', shape=style_shapes[i])
            net = vgg.net_preloaded(vgg_weights, image, pooling)
            style_pre = np.array([vgg.preprocess(styles[i], vgg_mean_pixel)])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))
                maps[i][layer] = features
                gram = np.matmul(features.T, features) / features.size
                style_features[i][layer] = gram

    initial_content_noise_coeff = 1.0 - initial_noiseblend

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        if initial is None:
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = tf.random_normal(shape) * 0.256
        else:
            initial = np.array([vgg.preprocess(initial, vgg_mean_pixel)])
            initial = initial.astype('float32')
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = (initial) * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (1.0 - initial_content_noise_coeff)
        image = tf.Variable(initial)
        net = vgg.net_preloaded(vgg_weights, image, pooling)

        # content loss
        content_layers_weights = {}
        content_layers_weights['relu4_2'] = content_weight_blend
        content_layers_weights['relu5_2'] = 1.0 - content_weight_blend

        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:
            # content_losses.append(binary_crossentropy(content_features[content_layer], net[content_layer]))
            content_losses.append(content_layers_weights[content_layer] * content_weight * tf.nn.l2_loss(
                    net[content_layer] - content_features[content_layer]) / content_features[content_layer].size)
        content_loss += reduce(tf.add, content_losses)

        #bias loss
        bias_loss = 0
        maps_bias = {}
        # maps_bias = [{} for _ in styles]
        # for i in range(len(styles)):
        bias_losses = []
        for layer in STYLE_LAYERS:
            # stderr.write('maps[i][layer]: ' + str(maps[i][layer].shape) + '\n')
            biases = []
            for i in range(len(styles)):
                bias = np.reshape(np.sum(maps[i][layer], axis=0), [1, -1])
                biases.append(bias)
            for i in range(bias.shape[1]):
                # stderr.write('shape: ' + str(bias.shape[1]) + '\n')
                k = random.randint(0, len(styles)-1)
                # stderr.write('k: ' + str(k) + ' i: ' + str(i) + '\n')
                bias[:,i] = biases[k][:,i];

            # stderr.write('bias.shape: ' + str(bias.shape) + '\n')
            assert bias.shape[0] == 1
            bias = np.tile(bias, [bias.shape[1], 1])
            assert bias.shape[0] == bias.shape[1]
            gram_bias = bias.T - bias
            maps_bias[layer] = gram_bias

            style_layer = net[layer]
            _, height, width, number = map(lambda i: i.value, style_layer.get_shape())
            size = height * width * number
            feats = tf.reshape(style_layer, (-1, number))
            feats = tf.reshape(tf.reduce_sum(feats, axis=0), [1, -1])
            feats = tf.tile(feats, [number, 1])
            feats_bias = tf.transpose(feats) - feats
            bias_losses.append(2 * tf.nn.l2_loss(feats_bias - gram_bias) / number / number)
        # bias_loss += style_weight * style_blend_weights[i] * reduce(tf.add, bias_losses)
        bias_loss += 1e-6 * reduce(tf.add, bias_losses)


        # style loss
        style_loss = 0
        # for i in range(len(styles)):
        #     style_losses = []
        #     for style_layer in STYLE_LAYERS:
        #         layer = net[style_layer]
        #         _, height, width, number = map(lambda i: i.value, layer.get_shape())
        #         size = height * width * number
        #         feats = tf.reshape(layer, (-1, number))
        #         gram = tf.matmul(tf.transpose(feats), feats) / size
        #         style_gram = style_features[i][style_layer]
        #         style_losses.append(style_layers_weights[style_layer] * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
        #         # style_losses.append(style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram))
        #     style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)

        # total variation denoising
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    tv_x_size))
        # overall loss
        # loss = content_loss
        # loss = bias_loss + content_loss + tv_loss
        loss = bias_loss + tv_loss
        # loss = style_loss + content_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

        def print_progress():
            stderr.write('  content loss: %g\n' % content_loss.eval())
            # stderr.write('    style loss: %g\n' % style_loss.eval())
            stderr.write('       tv loss: %g\n' % tv_loss.eval())
            stderr.write('     bias loss: %g\n' % bias_loss.eval())
            stderr.write('    total loss: %g\n' % loss.eval())

        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            # if (print_iterations and print_iterations != 0):
                # print_progress()
            for i in range(iterations):
                stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
                if(i % 100 == 0):
                    stderr.write('content loss: %g\n' % content_loss.eval())
                    stderr.write('bias loss: %g\n' % bias_loss.eval())
                    stderr.write('tv loss: %g\n' % tv_loss.eval())
                    stderr.write('total loss: %g\n' % loss.eval())
                    # stderr.write('bias loss: %g\n' % bias_loss.eval())
                train_step.run()

                last_step = (i == iterations - 1)
                if last_step or (print_iterations and i % print_iterations == 0):
                    print_progress()

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()

                    img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)
                    yield (
                        (None if last_step else i),
                        img_out
                    )


def binary_crossentropy(a, b):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    eps = 1e-8
    sig_a = sigmoid(a)
    sig_b = tf.sigmoid(b)
    kl = -( sig_a * tf.log(sig_b + eps) + (1.0 - sig_a) * tf.log(1.0 - sig_b + eps))
    return tf.reduce_sum(kl)

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb
