import numpy as np
import tflearn
import operator
from tflearn.data_preprocessing import ImagePreprocessing
from collections import deque

import cv2


"""
<< RealTimeEmotionDetection >>

This Class defines the methods necessary for real-time emotion detection of video captured through a webcam.
We use OpenCV's haar cascade classifier to recognize faces and crop the image of a face.
Then we take the face image and use our trained model to predict the most probable emotion associated with the image.
We also use a queue/deque to keep the 10 most recent emotions that are predicted and then choose the emotion with highest
incidence/probability to be printed to screen.

"""

class RealtimeEmotionDetection:

    def __init__(self):
        # We will be using a queue which stores the last n emotions detected, in order to get a more accurate prediction by smoothing
        self.emotions_deque = deque(maxlen=10)

    def emotion_smoothing(self, prediction):
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        emotion_values = {'Angry': 0.0, 'Disgust': 0.0, 'Fear': 0.0, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.0, 'Neutral': 0.0}

        # iterating over the predicted values output from trained model to get the max value and associated index
        emotion_prob_val, emotion_idx = max((val, idx) for (idx, val) in enumerate(prediction[0]))
        predicted_emotion = emotions[emotion_idx]

        # Push the latest emotion onto our deque and pop the oldest value (if any)
        self.emotions_deque.appendleft((emotion_prob_val, predicted_emotion))

        # go through the deque and extract emotion values to add to our emotion_values dictionary
        for data in self.emotions_deque:
            emotion_values[data[1]] += data[0]

        result = max(emotion_values.items(), key=operator.itemgetter(1))[0]
        return result

    def image_processing(self, roi_gray, img):
        img_scaled = np.array(cv2.resize(roi_gray, (48,48)), dtype=float)
        img_processed = img_scaled.flatten().reshape([-1, 48, 48, 1])

        #Using trained model to predict emotions from cropped image
        predicted_emotions = self.model.predict(img_processed)
        predicted_emotion = self.emotion_smoothing(predicted_emotions)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, "Current Detected Emotion: " + predicted_emotion, (50,450), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('img', img)

    def run(self):

        # Real-time pre-processing of the image data
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()

        # Real-time data augmentation
        img_aug = tflearn.ImageAugmentation()
        img_aug.add_random_flip_leftright()

        # Resnet model below:  Adapted from tflearn website
        self.n = 5 #32 layer resnet

        # Building Residual Network
        net = tflearn.input_data(shape=[None, 48, 48, 1], data_preprocessing=img_prep, data_augmentation=img_aug)
        net = tflearn.conv_2d(net, nb_filter=16, filter_size=3, regularizer='L2', weight_decay=0.0001)
        net = tflearn.residual_block(net, self.n, 16)
        net = tflearn.residual_block(net, 1, 32, downsample=True)
        net = tflearn.residual_block(net, self.n - 1, 32)
        net = tflearn.residual_block(net, 1, 64, downsample=True)
        net = tflearn.residual_block(net, self.n - 1, 64)
        net = tflearn.batch_normalization(net)
        net = tflearn.activation(net, 'relu')
        net = tflearn.global_avg_pool(net)

        # Regression
        net = tflearn.fully_connected(net, 7, activation='softmax')
        mom = tflearn.Momentum(learning_rate=0.1, lr_decay=0.0001, decay_step=32000, staircase=True, momentum=0.9)
        net = tflearn.regression(net, optimizer=mom,
                                 loss='categorical_crossentropy')

        self.model = tflearn.DNN(net, checkpoint_path='models/model_resnet_emotion',
                            max_checkpoints=10, tensorboard_verbose=0,
                            clip_gradients=0.)

        self.model.load('model.tfl')

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)

        #Main Loop where we will be capturing live webcam feed, crop image and process the image for emotion recognition on trained model
        while True:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                self.image_processing(roi_gray, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if (__name__ == "__main__"):
    realtimeEmotionDetection = RealtimeEmotionDetection()
    realtimeEmotionDetection.run()
