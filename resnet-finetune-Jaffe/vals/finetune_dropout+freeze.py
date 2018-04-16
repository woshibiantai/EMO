# A large portion of this code comes from the tf-learn example page:
# https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_cifar10.py

import tflearn
import numpy as np
from numpy import genfromtxt
from tflearn.data_preprocessing import ImagePreprocessing


if __name__ == "__main__":
    n = 5
    print("loading data")
    #Data loading and pre-processing
    X = np.asarray(genfromtxt('data/training_data.csv', delimiter=' ',  skip_header=1,  dtype=float))
    Y = np.asarray(genfromtxt('data/training_labels.csv', delimiter=' ', skip_header=1, dtype=int))

    X_test = np.asarray(genfromtxt('data/test_data.csv', delimiter=' ',  skip_header=1,  dtype=float))
    Y_test = np.asarray(genfromtxt('data/test_labels.csv', delimiter=' ', skip_header=1, dtype=int))
    predict_value = np.asarray(genfromtxt('data/test_image.csv', delimiter=' ', dtype=float))

    print("done loading data")
    print("reshaping images")
    predict_value = predict_value.reshape([-1, 48, 48, 1])

    # Reshape the images into 48x4
    X = X.reshape([-1, 48, 48, 1])
    X_test = X_test.reshape([-1, 48, 48, 1])

    print("one-hot encoding the labels")
    # One hot encode the labels
    Y = tflearn.data_utils.to_categorical(Y, 7)
    Y_test = tflearn.data_utils.to_categorical(Y_test, 7)

    print("real time image processing of image data")
    # Real-time preprocessing of the image data
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    print("real time data augmentation")
    # Real-time data augmentation
    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_flip_leftright()

    print("building ResNet")
    # Building Residual Network
    net = tflearn.input_data(shape=[None, 48, 48, 1], data_preprocessing=img_prep, data_augmentation=img_aug)
    net = tflearn.conv_2d(net, nb_filter=16, filter_size=3, regularizer='L2', weight_decay=0.0001)
    net = tflearn.residual_block(net, n, 16)
    net = tflearn.dropout(net, 0.8)
    net = tflearn.residual_block(net, 1, 32, downsample=True)
    net = tflearn.dropout(net, 0.8)
    net = tflearn.residual_block(net, n-1, 32)
    net = tflearn.dropout(net, 0.8)
    net = tflearn.residual_block(net, 1, 64, downsample=True)
    net = tflearn.dropout(net, 0.8)
    net = tflearn.residual_block(net, n-1, 64)
    net = tflearn.dropout(net, 0.8)
    net = tflearn.batch_normalization(net)
    net = tflearn.dropout(net, 0.8)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    net = tflearn.dropout(net, 0.8)

    print("regression....")
    # Regression
    net = tflearn.fully_connected(net, 7, activation='softmax', restore=False)
    mom = tflearn.Momentum(learning_rate=0.1, lr_decay=0.0001, decay_step=32000, staircase=True, momentum=0.9)
    net = tflearn.regression(net, optimizer=mom,
                             loss='categorical_crossentropy')

    print("training..........")
    # Training
    model = tflearn.DNN(net, checkpoint_path='jaffe-model',
                        max_checkpoints=20, tensorboard_verbose=0,
                        clip_gradients=0.)

    model.load('model.tfl')

    model.fit(X, Y, n_epoch=150, snapshot_epoch=False, snapshot_step=500,
              show_metric=True, batch_size=128, shuffle=True, run_id='resnet_emotion')

    print("evaluating...................")
    score = model.evaluate(X_test, Y_test)
    print('Test accuracy: ', score)

    model.save('model.tfl')
    #prediction = model.predict(predict_value)
    #print(prediction)
