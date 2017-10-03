import keras
from dual_im_gen import dual_im_gen
from models import dual_im

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import metrics
from sklearn.metrics import confusion_matrix, f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model


model = dual_im(24, 'mixed0', 'mixed1')

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=[
        metrics.categorical_accuracy,
        metrics.top_k_categorical_accuracy,
        #top3_acc
        ]
    )



dim_gen2 = dual_im_gen("/Users/user1/Documents/thesis/data/split_images/benthoz_retrain/testing", 32)
dim_gen = dual_im_gen("/Users/user1/Documents/thesis/data/split_images/benthoz_retrain/training", 32)

history_tl = model.fit_generator(
    dim_gen,
    epochs=1,
    steps_per_epoch=10,
    validation_data=dim_gen2,
    validation_steps=4,
    class_weight='auto',
    #callbacks = tensorboard,
    verbose =2
    )
