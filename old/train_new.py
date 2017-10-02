'''
Train a new instance of a tensorflow model
Jayant Khatkar
'''
import sys
import os
model_locations = os.path.abspath(os.path.join(os.path.curdir, 'Models'))
sys.path.insert(0, model_locations)
#other_locations = os.path.abspath(os.path.join(os.path.curdir, '..'))
#sys.path.insert(0, other_locations)
import tensorflow as tf
print("tensorflow version")
print(tf.__version__)

import numpy as np
from data_gen import *
import time
from   sklearn.metrics import confusion_matrix, f1_score

#load model(s)
from CNN2 import *
from test import *
from restorer import *

def train_model(train_path, dev_path, test_path, save_path, num_epochs, batch_size, input_dim, out_dim, save_freq):

    x_in  = tf.placeholder(
        tf.float32,
        [None]+input_dim,
        name="x_in"
        )

    target = tf.placeholder(
        tf.uint8,
        [None],
        name = "target"
        )

    model = CNN2(
        x_in,
        target,
        input_dim[0],
        n_classes=out_dim
        )

    #print model details for the log
    model.info()

    # save directeries
    tensorboard_path, model_path = save_paths(
        save_path,
        model.model_name,
        )


    sess  = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(
        tensorboard_path,
        graph=tf.get_default_graph()
        )


    print('\nTraining Model...\n')

    # generate batches
    batches_train = batch_image_generator(train_path, batch_size, num_epochs)
    print('\nDebug: Successfully created batch generator...\n')
    best_f1=0
    iter_index = 0

    #train on batches
    for batch_x, batch_y in batches_train:
        #s_t = time.time()
        _ , summary = sess.run(
            [model.optimize, model.tensorboard_summary],
            feed_dict={
                x_in:   batch_x,
                target: batch_y
                }
            )
        #e_t = time.time()
        if iter_index==0:
            print('\nDebug: Successfully trained one interation...\n')
        #print('time taken = ' + str(e_t-s_t) + 's')

        #tensorboard summary
        summary_writer.add_summary(summary, iter_index)

        #if iter_index==0:
        #    print('\nDebug: Successfully added summary...\n')
        #print(iter_index)
        iter_index += 1

        if iter_index%save_freq==0:
            #calculate train & dev performance after training
            print('\nDebug: Successfully trained n interation, testing...\n')
            #trn_acc, trn_f1, trn_cm = test_model(train_path, model, sess, out_dim)
            #accuracy, cm, y_pred, y_real, labels
            dev_acc, dev_cm, yp, yr, labs = test_model(dev_path,   model, sess,out_dim)
            display_epoch_performance(iter_index, dev_acc, dev_cm, "dev")
            #display_epoch_performance(iter_index, trn_acc, trn_f1, trn_cm, "trn")
            print('\nDebug: Successfully tested...\n')

        	#if improved performance, then save model
            if best_f1 < dev_acc:
                best_f1 = dev_acc 
                _ = saver.save(sess, model_path)
                print('\nDebug: Successfully saved model...\n')
                #print('ENDING')
                #sys.exit()

    print("Completed training model.")
    #test acter completing training
    accuracy, cm, yp, yr, labs = test_model(test_path, model, sess,out_dim)
    display_epoch_performance(iter_index, accuracy, cm, "test")
    np.savez(os.path.join(tensorboard_path,'vars.npz'), conf_m = cm,y_pred =  yp, y_real = yr, labels = labs)
# /project/RDS-FEI-BEN_DL-RW/
# /project/RDS-FEI-BEN_DL-RW/split_images/benthoz_split
if __name__ == '__main__':
    flags = tf.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string(
    'train_data',
    '/project/RDS-FEI-BEN_DL-RW/split_images/benthoz_split/training',
    'Folder on gcloud where training data (images) is stored, including labels.csv'
    )
    flags.DEFINE_string(
    'validation_data',
    '/project/RDS-FEI-BEN_DL-RW/split_images/benthoz_split/valid',
    'Folder on gcloud where validation data (images) is stored, including labels.csv'
    )
    flags.DEFINE_string(
    'test_data',
    '/project/RDS-FEI-BEN_DL-RW/split_images/benthoz_split/test',
    'Folder on gcloud where test data (images) is stored, including labels.csv'
    )
    flags.DEFINE_string(
    'save_path',
    '/project/RDS-FEI-BEN_DL-RW/output',
    'Folder on gcloud where validation data (images) is stored, including labels.csv'
    )
    flags.DEFINE_integer(
    'epochs',
    10,
    'Number of epochs'
    )
    flags.DEFINE_integer(
    'batch_size',
    32,
    'c\'mon'
    )
    flags.DEFINE_integer(
    'save_freq',
    5000,
    'Number of iterations between checking if model has improved or not (and saving if it has)'
    )
    flags.DEFINE_integer(
    'im_size',
    100,
    'square images of side length im_size'
    )
    flags.DEFINE_integer(
    'out_size',
    24,
    'number of categories'
    )
    

    train_path      = FLAGS.train_data
    dev_path        = FLAGS.validation_data
    test_path       = FLAGS.test_data
    save_path       = FLAGS.save_path
    num_epochs      = FLAGS.epochs
    batch_size      = FLAGS.batch_size
    save_freq       = FLAGS.save_freq
    input_dim       = [FLAGS.im_size,FLAGS.im_size,3]
    out_dim         = FLAGS.out_size
    train_model(train_path,dev_path,test_path,save_path,num_epochs, batch_size,input_dim,out_dim,save_freq)
