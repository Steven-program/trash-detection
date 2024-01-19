import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
import cv2 as cv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# LOAD DATA
DIR = ""
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DIR,
    validation_split=0.1,
    subset="training",
    seed=42,
    batch_size=128,
    smart_resize=True,
    image_size=(256, 256),
)
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DIR,
    validation_split=0.1,
    subset="validation",
    seed=42,
    batch_size=128,
    smart_resize=True,
    image_size=(256, 256),
)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

numClasses = len(train_dataset.class_names)

baseModel = tf.keras.applications.MobileNetV3Large(
    input_shape=(256, 256, 3), weights="imagenet", include_top=False, classes=numClasses
)
for layers in baseModel.layers[:-6]:
    layers.trainable = False

last_output = baseModel.layers[-1].output
x = tf.keras.layers.Dropout(0.45)(last_output)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(
    256,
    activation=tf.keras.activations.elu,
    kernel_regularizer=tf.keras.regularizers.l1(0.045),
    activity_regularizer=tf.keras.regularizers.l1(0.045),
    kernel_initializer="he_normal",
)(x)
x = tf.keras.layers.Dropout(0.45)(x)
x = tf.keras.layers.Dense(numClasses, activation="softmax")(x)

model = tf.keras.Model(inputs=baseModel.input, outputs=x)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00125),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

epochs = 50
lrCallback = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-3 * 10 ** (epoch / 30)
)
stepDecay = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 0.1 * 0.1 ** math.floor(epoch / 6)
)
history = model.fit(
    train_dataset, validation_data=test_dataset, epochs=epochs, callbacks=[]
)


# Load your machine learning model
model = tf.keras.models.load_model("/Users/rene/Desktop/model.h5")

classes = [
    "Aluminium",
    "Carton",
    "Glass",
    "Organic Waste",
    "Other Plastics",
    "Paper and Cardboard",
    "Plastic",
    "Textiles",
    "Wood",
]

# Create an image data generator to normalize and augment the images
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)

# Start the camera capture loop
cap = cv2.VideoCapture(1)

while True:
    # Capture an image from the webcam
    ret, frame = cap.read()

    # Resize the image to match the model's input size
    frame = cv2.resize(frame, (256, 256))

    # Normalize and augment the image
    frame = image_generator.random_transform(frame)

    # Convert the image to HSV encoding
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Preprocess the image
    img_array = tf.keras.preprocessing.image.img_to_array(frame)
    img_array = tf.expand_dims(img_array, 0)

    # Make a prediction
    predictions = model.predict(img_array)

    # Get the predicted class label
    class_label = np.argmax(predictions)

    # Label the image based on the predicted class label
    label = classes[class_label]

    # Display the predicted class label on the image
    cv2.putText(frame, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the image
    cv2.imshow("frame", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


def plot_confusion_matrix(cm, target_names, cmap=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title("Confusion matrix")
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, "{:,}".format(cm[i, j]), horizontalalignment="center", color="black"
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel(
        "Predicted label\naccuracy={:0.4f}%; misclass={:0.4f}%".format(
            accuracy, misclass
        )
    )
    plt.show()


plt.figure(figsize=(10, 10))
true = []
predictions = []

"""
for images, labels in test_dataset.take(50):
  pred = model.predict(images)
  for i in range(32):
    try:
      ax = plt.subplot(4, 8, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      #print(classes[np.argmax(pred[i])], 100 * np.max(pred[i]), "real = " + str(classes[labels[i]]))

      true.append(labels[i])
      predictions.append(np.argmax(pred[i]))

      plt.title(classes[labels[i]])
      plt.axis("off")
    except:
      print()

"""
path = "WasteImagesDataset"
for i in os.listdir(path):
    folderPath = os.path.join(path, i)
    for j in os.listdir(folderPath)[:550]:
        fullPath = os.path.join(folderPath, j)
        try:
            img = tf.keras.preprocessing.image.load_img(
                fullPath, target_size=(256, 256)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            preds = model.predict(img_array)
            true.append(classes.index(i))
            predictions.append(np.argmax(preds))
        except:
            print("Error on image:", fullPath)

plot_confusion_matrix(tf.math.confusion_matrix(true, predictions), classes)
