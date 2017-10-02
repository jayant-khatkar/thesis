import numpy as np
import pandas as pd
#import cv2
import tensorflow as tf
from scipy import misc
import os
import time
import sys





def image_address(data, index):
    web = data.web_location[index].split('/')
    return './benthoz2015/'+ web[0] +'/'+ web[-1]

def get_annotations(location):
    folder = './benthoz2015/' + location
    files_in_folder = [each for each in os.listdir(folder) if each.endswith('.csv')]
    #just read first csv in the folder (there should only be 1)
    file_path = './benthoz2015/' + location +'/'+ files_in_folder[0]
    return pd.read_csv(file_path)

def get_image_annotations(annotations, image_id):
    return annotations.loc[annotations['image__id'] == image_id]

# def draw_annotations(img_original, img_ann):
#     img = img_original.copy()
#     for i in img_ann.index:
#         pixel = (int(img_ann.at[i,'x']*img.shape[1]),int(img_ann.at[i,'y']*img.shape[0]))
#         cv2.circle(img, pixel, 5, (255,0,0), 4)
#     return img

def crop_annotation(img, img_ann, i, box_size):
    center_pixel = (int(img_ann.at[i,'x']*img.shape[1]),int(img_ann.at[i,'y']*img.shape[0]))

    startx = int(center_pixel[0]-np.floor(box_size/2))
    endx   = int(center_pixel[0]+np.ceil(box_size/2))
    starty = int(center_pixel[1]-np.floor(box_size/2))
    endy   = int(center_pixel[1]+np.ceil(box_size/2))

    if startx<0 or endx>=img.shape[1] or starty<0 or endy>=img.shape[0]:
        val = 0
        #print("Box goes off image")
    else:
        val = img[starty:endy,startx:endx]
    return val

def get_image_indices_from_location(location, imagelist):
    indices = []
    for i in imagelist.index.values:
        if imagelist.web_location[i].split('/')[0]==location:
            indices.append(i)
    return indices

def crop_image_annotations(img_index, box_size, imagelist):

    location = imagelist.web_location[img_index].split('/')[0]
    annotations = get_annotations(location)

    img_ann = get_image_annotations(annotations, int(imagelist.image__id[img_index]))

    if len(img_ann)==0:
        print("no annotations on image " + str(imagelist.image__id[img_index]))

    img = misc.imread(image_address(imagelist,img_index))
    crops=[]

    for i in img_ann.index.values:
        crops.append(crop_annotation(img, img_ann, i, box_size))

    return crops, img_ann

def crop_multiple_images(img_indices, box_size, imagelist):

    #location = imagelist.web_location[img_index].split('/')[0]

    crops, annotations = crop_image_annotations(img_indices[0], box_size, imagelist)

    for i in range(1,len(img_indices)):
        crops_i, img_ann_i = crop_image_annotations(img_indices[i], box_size, imagelist)
        crops += crops_i
        annotations = pd.concat([annotations,img_ann_i])

    return crops, annotations

def save_crops_old(crops, annotations, save_location):
#deprecated
    if not os.path.isdir(save_location):
        os.makedirs(save_location)

    f_names = pd.DataFrame(save_location + '/' + annotations['image__id'].apply(str) + '-' + annotations['Unnamed: 0'].apply(str) + '.png')
    f_names.columns=['filename']
    for i, j in zip(annotations.index.values, range(0,len(annotations.index.values))):
        file_name = f_names['filename'][i]

        if type(crops[j]) is not int:
            misc.imsave(file_name, crops[j])

    save_csv = pd.concat([f_names['filename'], annotations['label_id']], axis=1)
    save_csv.to_csv(save_location +'/'+ 'labels.csv')

def crop_and_save_all_from(location, save_location, box_size, imagelist):
#deprecated
    indicies = get_image_indices_from_location(location, imagelist)
    crops, img_ann = crop_multiple_images(indicies, box_size, imagelist)

    save_crops_old(crops, img_ann, save_location)


def extract_annotations_from(location, save_location, box_size, imagelist):

    #images to use
    img_indices = get_image_indices_from_location(location, imagelist)

    #creat save driectory
    if not os.path.isdir(save_location):
        os.makedirs(save_location)

    save_csv = pd.DataFrame()
    #for each image
    for i in range(len(img_indices)):
        crops_i, img_ann_i = crop_image_annotations(img_indices[i], box_size, imagelist)

        #labels for crops
        f_names = pd.DataFrame(save_location + '/' + img_ann_i['image__id'].apply(str) + '-' + img_ann_i['Unnamed: 0'].apply(str) + '.png')
        f_names.columns=['filename']
        image_label = pd.concat([f_names['filename'], img_ann_i['label_id']], axis=1)


        #save each crop (and remove edge annotations)
        for i, j in zip(img_ann_i.index.values, range(len(img_ann_i.index.values))):
            file_name = image_label['filename'][i]

            if type(crops_i[j]) is not int:
                misc.imsave(file_name, crops_i[j])
            else:
                image_label = image_label.drop(i)

        #labels


        #append to all image labels
        save_csv = pd.concat([save_csv, image_label])
    save_csv.to_csv(save_location +'/'+ location + '-labels.csv')



#im_list = create_split_dataset(imagelist, save='data/benthoz2015/imagelist_split.csv')
def create_split_dataset(imagelist, training = 0.7, validation=0.1, test=0.2, save = None ):

    total_images = len(imagelist)

    train_n = np.floor(total_images*training).astype('int')
    eval_n  = np.floor(total_images*validation).astype('int')
    test_n  = np.floor(total_images*test).astype('int')

    indicies = np.random.permutation(total_images).astype('int')

    train_indices = indicies[0:train_n]
    eval_indices  = indicies[train_n:train_n+eval_n]
    test_indices = indicies[train_n+eval_n:train_n+eval_n+test_n]

    sets = np.zeros(total_images) #0->training set
    sets[eval_indices] = 1        #1->evaluation set
    sets[test_indices] = 2        #2_>test set
    set_col = pd.DataFrame(sets.astype('int'),columns=['sets'])
    im_list = pd.concat([imagelist,set_col],axis=1)

    if save is not None:
        im_list.to_csv(save,  index=False)

    return im_list




def extract_n_annotations(n_images, imagelist, save_location, box_size=100 , set_type='train'):

    if set_type == 'train':
        set_type=0
    elif set_type == 'eval':
        set_type=1
    else:
        set_type=2

    #narrow down list of images ebing used
    imagelist2 = imagelist.loc[imagelist['sets'] == set_type]

    if n_images==0:
        n_images = len(imagelist2.index.values)
        print("total of " + str(n_images) + " to extract")
    #images to use
    img_indices = np.random.choice(imagelist2.index.values,n_images, replace=False)

    #creat save driectory
    if not os.path.isdir(save_location):
        os.makedirs(save_location)

    save_csv = pd.DataFrame()
    #for each image
    for i in range(len(img_indices)):
        print(i)
        crops_i, img_ann_i = crop_image_annotations(img_indices[i], box_size, imagelist)

        #labels for crops
        f_names = pd.DataFrame(save_location + '/' + img_ann_i['image__id'].apply(str) + '-' + img_ann_i['Unnamed: 0'].apply(str) + '.png')
        f_names.columns=['filename']
        image_label = pd.concat([f_names['filename'], img_ann_i['label_id']], axis=1)


        #save each crop (and remove edge annotations)
        for i, j in zip(img_ann_i.index.values, range(len(img_ann_i.index.values))):
            file_name = image_label['filename'][i]

            if type(crops_i[j]) is not int:
                misc.imsave(file_name, crops_i[j])
            else:
                image_label = image_label.drop(i)

        #labels


        #append to all image labels
        save_csv = pd.concat([save_csv, image_label])
    save_csv.to_csv(save_location +'/labels.csv')



if __name__ == '__main__':
    #file_path = 'data/benthoz2015/BENTHOZ-2015-imagelist.csv'


    locations = ['Batemans201011','Batemans201211',
    'PS201012','PS201211','SEQueensland201010',
    'SolitaryIs201208','Tasmania200810','Tasmania200903',
    'Tasmania200906','WA201104','WA201204','WA201304']

    flags = tf.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string(
    'imagelist_location',
    './benthoz2015/imagelist_split.csv',
    'File on gcloud where image labels are stored (with labels for which set they are in)'
    )

    flags.DEFINE_string(
    'output_location',
    './benthoz_split/',
    'File on gcloud where image labels are stored (with labels for which set they are in)'
    )

    flags.DEFINE_integer(
    'box_size',
    100,
    'square size of annotations'
    )

    file_path = FLAGS.imagelist_location
    imagelist = pd.read_csv(file_path)

    box_size = FLAGS.box_size
    save_location = FLAGS.output_location

    print('extracting training data')
    extract_n_annotations(0, imagelist, save_location+'training/', box_size=box_size , set_type='train')

    #print('extracting validation data')
    #extract_n_annotations(0, imagelist, save_location+'valid/', box_size=box_size , set_type='eval')

    #print('extracting test data')
    #extract_n_annotations(0, imagelist, save_location+'test/', box_size=box_size , set_type='test')


    #create_split_dataset(imagelist, training = 0.7, validation=0.1, test=0.2, save = save_path )


    #s_t = time.time()
    #extract_n_annotations(n_images, imagelist, save_location, box_size=box_size , set_type=save_location)
    #extract_n_annotations(100, imagelist, save_location, box_size=100 , set_type='train')
    #e_t = time.time()
    #print('time taken = ' + str(e_t-s_t) + 's')
