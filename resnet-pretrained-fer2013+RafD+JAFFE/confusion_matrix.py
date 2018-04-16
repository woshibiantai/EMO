import tflearn
import numpy as np
from numpy import genfromtxt
from tflearn.data_preprocessing import ImagePreprocessing
import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Emotion Prediction Confusion Matrix")
    else:
        print('Emotion Prediction Confusion Matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":
    n = 5

    X_test = np.asarray(genfromtxt('Custom_Emotions_Test_Data/test_data.csv', delimiter=' ',  skip_header=1,  dtype=float))
    Y_true = np.asarray(genfromtxt('Custom_Emotions_Test_Data/test_labels.csv', delimiter=' ', skip_header=1, dtype=int))
    Y_test = np.asarray(genfromtxt('Custom_Emotions_Test_Data/test_labels.csv', delimiter=' ', skip_header=1, dtype=int))

    X_test = X_test.reshape([-1, 48, 48, 1])
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

    net = tflearn.input_data(shape=[None, 48, 48, 1], data_preprocessing=img_prep, data_augmentation=img_aug)
    net = tflearn.conv_2d(net, nb_filter=16, filter_size=3, regularizer='L2', weight_decay=0.0001)
    net = tflearn.residual_block(net, n, 16)
    net = tflearn.residual_block(net, 1, 32, downsample=True)
    net = tflearn.residual_block(net, n-1, 32)
    net = tflearn.residual_block(net, 1, 64, downsample=True)
    net = tflearn.residual_block(net, n-1, 64)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)

    net = tflearn.fully_connected(net, 7, activation='softmax')
    mom = tflearn.Momentum(learning_rate=0.1, lr_decay=0.0001, decay_step=32000, staircase=True, momentum=0.9)
    net = tflearn.regression(net, optimizer=mom,
                            loss='categorical_crossentropy')

    model = tflearn.DNN(net, checkpoint_path='models/model_resnet_emotion',
                        max_checkpoints=20, tensorboard_verbose=0,
                        clip_gradients=0.)

    #loading trained model
    model.load('fer2013-rafd-jaffe-model/model.tfl')

    #testing the accuracy of the trained model
    print("evaluating...................")
    score = model.evaluate(X_test, Y_test)
    print ('Test accuracy: ', score)

    prediction = model.predict(X_test)

    #convert probability vector to class vector (rmb to use the class vector not the one hot label vector for y-test)
    Y_predicted = np.argmax(prediction, axis=1)
    print ('Correct Labels:', Y_true)
    print ('Pred Labels:', Y_predicted)

    #labels for confusion matrix
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    cnf_matrix = confusion_matrix(Y_true, Y_predicted)
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cnf_matrix, classes=emotions, normalize=True,
                      title='Normalized confusion matrix')

    plt.show()
