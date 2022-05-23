import tensorflow as tf
import cv2 as cv
import numpy as np
from time import time
from os import listdir


def read_image(capture, bool_ret=False):

    """
    Reads a image (a video frame) from a VideoCapture object.
    Modifies its color format from BGR to RGB. Then, applies
    image preprocessing to prepare input image for model input.

    inputs:
        capture - opencv video capture object
        bool_ret - a switch that can determine the num. of returning
        variables. adds boolean state of captured image to returning
        variables.

    outputs:
        image - captured input image (frame)
        bool_image - (optional) boolean state captured input image
    """

    # read the image frame
    bool_image, image = capture.read()
    # transform the color channel format from BGR to RGB
    # image = cv.cvtColor(image, cv.COLOR_BAYER_BG2RGB)

    # return image and bool state
    if bool_ret:
        return image, bool_image
    # return just the image
    return image


def preprocess_image(image, size):

    """
    Takes an input image, turns it into a tensor, applies resizing and
    scaling (normalizing)respectively to input image. Returns processed
    image.

    inputs:
        image - input image to process

    outputs:
        image - processed image
    """

    # convert the array to tensor
    image = tf.constant(image)
    # resizing the input image
    image = tf.image.resize(image, [size, size])
    # scaling the input image
    image = tf.keras.layers.Rescaling(scale=1.0 / 255)(image)
    # return processed image
    return image


def get_classes_list(root_path):

    # get class names as list
    classes = listdir(root_path)
    # return classes list
    return classes


def predict_image(model, image):

    # add batch axis to fit predict method
    image = tf.expand_dims(image, axis=0)
    # predict the current frame
    pred = model.predict(image)
    # return the prediction
    return pred


def render_result(image, pred, classes):

    # get index of max. probability class
    label_ind = tf.argmax(pred).numpy()
    # get probability value of the max. prob. class
    probability = pred.numpy()[label_ind]
    # get the class label
    label = classes[label_ind]
    # convert the image from tensor to numpy array form
    image = cv.cvtColor(image.numpy(), cv.COLOR_RGB2BGR)
    # write the label and probability value on the image
    cv.putText
    # render the result image
    cv.imshow("Prediction Screen", image)
    # return literally anything
    return "I'm done, Boss!"


#### If __name__= main

# GPU Configurations
gpus = tf.config.list_physical_devices("GPU")
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(e)


# defining video capture object
obj_capture = cv.VideoCapture(0)

# importing TF2 model
model_name = "MLSRNet_AlexNet_v5"
path_models_root = "/home/deniz/Desktop/CODE-ENV/image-classifier-with-Keras/models/"
path_model = path_models_root + model_name
model = tf.keras.models.load_model(path_model)

while True:

    # get RGB-formatted input image
    image_original = read_image(obj_capture)

    # apply preprocessing to input image
    image_processed = preprocess_image(image_original, 227)
    # print(image_processed)

    # make predict on the image
    # prediction = predict_image(model, image_procqessed)

    cv.imshow("test", image_original)
    # render the result
    # flag = render_result(image_original, prediction, classes)

    # break-the-loop check with 25 ms delay
    if cv.waitKey(25) & 0xFF == ord("q"):
        break

obj_capture.release()
cv.destroyAllWindows()
