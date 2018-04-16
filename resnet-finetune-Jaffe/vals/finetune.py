import tensorflow as tf
import tflearn
import numpy as np
from numpy import genfromtxt
from tflearn.data_preprocessing import ImagePreprocessing


if __name__ == "__main__":
    n = 5
    print("loading data")
    # Data loading and pre-processing
    X = np.asarray(genfromtxt('JAFFE-data/training_data.csv', delimiter=' ',  skip_header=1,  dtype=float))
    Y = np.asarray(genfromtxt('JAFFE-data/new_training_labels.csv', delimiter=' ', skip_header=1, dtype=int))

    # X_test = np.asarray(genfromtxt('data/Test_Data.csv', delimiter=' ',  skip_header=1,  dtype=float))
    # Y_test = np.asarray(genfromtxt('data/Test_Labels.csv', delimiter=' ', skip_header=1, dtype=int))
    # predict_value = np.asarray(genfromtxt('data/test_image.csv', delimiter=' ', dtype=float))

    print("done loading data")
    print("reshaping images")
    # predict_value = predict_value.reshape([-1, 48, 48, 1])

    # Reshape the images into 48x4
    X = X.reshape([-1, 48, 48, 1])
    # X_test = X_test.reshape([-1, 48, 48, 1])

    print("one-hot encoding the labels")
    # One hot encode the labels
    Y = tflearn.data_utils.to_categorical(Y, 4)
    # Y_test = tflearn.data_utils.to_categorical(Y_test, 7)

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
    # Building Residual Network for Fine tuning
    # v1: freeze conv and residual layer + softmax layer classify to 3 classes: Positive, Neutral, Negative

    net = tflearn.input_data(shape=[None, 48, 48, 1], data_preprocessing=img_prep, data_augmentation=img_aug)
    net = tflearn.conv_2d(net, nb_filter=16, filter_size=3, regularizer='L2', weight_decay=0.0001, trainable=False)
    net = tflearn.residual_block(net, n, 16, trainable=False)
    net = tflearn.residual_block(net, 1, 32, downsample=True, trainable=False)
    net = tflearn.residual_block(net, n-1, 32, trainable=False)
    net = tflearn.residual_block(net, 1, 64, downsample=True, trainable=False)
    net = tflearn.residual_block(net, n-1, 64, trainable=False)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    # Regression
    net = tflearn.fully_connected(net, 4, activation='softmax', restore=False)
    mom = tflearn.Momentum(learning_rate=0.0001, lr_decay=0.7, decay_step=10, staircase=True, momentum=0.9)
    net = tflearn.regression(net, optimizer=mom,
                             loss='categorical_crossentropy')

    # TO DO:
    # find out best optimiser
    # find out best learning rate, learning decay, and decay step for fine tuning

    # find out how to write to tensorboard

    # sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
    # net = tflearn.regression(softmax, optimizer=sgd, metric='accuracy', loss='categorical_crossentropy')

    # file_writer = tf.summary.FileWriter('/logs', sess.graph)

    print("training..........")
    # Training (Fine-tuning)
    model = tflearn.DNN(net, checkpoint_path='finetuned-jaffe-models-v1/model_resnet_jaffe',
                        max_checkpoints=20, tensorboard_verbose=0, tensorboard_dir="logs", 
                        clip_gradients=0.)

    model.load('fer2013-rafd-model/model.tfl')

    model.fit(X, Y, n_epoch=10, validation_set=0.1, snapshot_epoch=False, snapshot_step=200,
              show_metric=True, batch_size=10, shuffle=True, run_id='resnet_emotion_JAFFE')

    # print("evaluating...................")
    # score = model.evaluate(X_test, Y_test)
    # print('Test accuracy after fine-tuning: ', score)

    model.save('jaffe-models/jaffe.modelv1.tfl')

    print('done')
    # prediction = model.predict(X_test)

    # #convert probability vector to class vector (rmb to use the class vector not the one hot label vector for y-test)
    # Y_predicted = np.argmax(prediction, axis=1)
    # print ('Correct Labels:', Y_true)
    # print ('Pred Labels:', Y_predicted)

    # #labels for confusion matrix
    # emotions = ["Positive", "Negative", "Neutral"]

    # cnf_matrix = confusion_matrix(Y_true, Y_predicted)
    # np.set_printoptions(precision=2)
    # plot_confusion_matrix(cnf_matrix, classes=emotions, normalize=True,
    #                   title='Normalized confusion matrix')

    # plt.show()


