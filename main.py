from cProfile import label
from wsgiref.validate import PartialIteratorWrapper
import cv2
import tensorflow
import tensorflow.compat.v1 as tf
import tensorflow.keras
import matplotlib.pyplot as plt
import time
import os
import uuid
import json
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split
import albumentations as alb
# The following imports are mainly for building the model->
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D,Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16
# print("Changing tensorflow nature from 2.x to 1.x")
# tf.disable_v2_behavior()

print("All packages imported")

# # Setting up the required paths
paths = {
    "DATA_PATH":"data",
    "IMAGES_PATH":os.path.join("data","images"),
    "LABEL_PATH":os.path.join("data","labels")
}
for folders in paths.values():
    if not os.path.exists(folders):
        os.mkdir(folders)
        print(f"{folders} established")
    else:
        print("The paths are already established")

# # Collecting the images samples:
num_img = 30
def collect_images_sample():
    cap = cv2.VideoCapture(1)
    for num in range(num_img):
        print(f"Collected {num} images")
        ret,frame = cap.read()
        imgname = os.path.join(paths["IMAGES_PATH"],f"{str(uuid.uuid1())}.jpg")
        cv2.imwrite(imgname, frame)
        cv2.imshow("frame",frame)
        time.sleep(0.5)

        if cv2.waitKey(1) & 0xFF==ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

sample_collection = input("Do you want to collect images samples?\n").lower()

if sample_collection=="yes":
    collect_images_sample()

# # Avoiding getting too many OOM errors by limiting GPU growth->
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

print(tf.config.list_physical_devices('GPU'))
# # Loading images into data pipeline:

images = tf.data.Dataset.list_files('data\\images\\*jpg', shuffle=False)
# print(images.as_numpy_iterator().next())

def load_image(x):
    byte_image = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_image)
    return img

images = images.map(load_image)
print(images.as_numpy_iterator().next())
print(type(images))

# Visualizing the images-
image_generator = images.batch(6).as_numpy_iterator() #Shall pack ourimages in the batch and then show us together the num of images specified in the funcction
plot_images = image_generator.next()

fig,ax  = plt.subplots(ncols=6,figsize=(20,20))
for idx,image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()

# Automatically splitting the training and testng data-
# Pushing all the items which we have in a list-
all_image_data = []
all_label_data = []
partition_success=False

for items in os.listdir(paths["IMAGES_PATH"]):
    all_image_data.append(items)
for labels in os.listdir(paths["LABEL_PATH"]):
    all_label_data.append(labels)

try:
    image_train,image_test,label_train,label_test = train_test_split(all_image_data,all_label_data)
    partition_success=True
    print("Data partition succcessful")
except Exception as e:
    print(f"Data partition failed due to {e}")

print("Printing the items in image_train->") #All this code was just for tesing and visualization purposes
for items in image_train:
    print(items)
    print(os.path.dirname(os.path.realpath(items)))

partitoned_data_paths = {
    "TRAINING_IMAGES":os.path.join("data","images","train"),
    "TESTING_IMAGES":os.path.join("data","images","test"),
    "TRAINING_LABELS":os.path.join("data","labels","train"),
    "TESTING_LABELS":os.path.join("data","labels","test")
}

partitioning_paths = []
def establish_partition_paths(partitoned_data_paths):
    for folders in partitoned_data_paths.values():
        if not os.path.exists(folders):
            os.mkdir(folders)
            print(f"{folders} established")
            print(os.path.dirname(os.path.realpath(folders)))
            partitioning_paths.append(folders)
        else:
            print(f"{folders} already established")
            print(os.path.dirname(os.path.realpath(folders)))

print("Printing the folders established for data partitioning")
for items in partitioning_paths:
    print(items)
    print(os.path.dirname(os.path.realpath(items))+items)
     

if partition_success==True:
    establish_partition_paths(partitoned_data_paths=partitoned_data_paths)

def move_partitioned_data(training_images,testing_images,training_labels,testing_labels):
    for folders in ["train","test"]:
        if folders=="train":
            for files in training_labels:
                existing_file_path = os.path.join("data","labels",files)
                if os.path.exists(existing_file_path):
                    new_file_path = os.path.join("data",folders,"labels",files)
                    os.mkdir(new_file_path)
                    os.replace(existing_file_path,new_file_path)
            for files in training_images:
                existing_file_path = os.path.join("data","images",files)
                if os.path.exists(existing_file_path):
                    new_file_path = os.path.join("data",folders,"images",files)
                    os.mkdir(new_file_path)
                    os.replace(existing_file_path,new_file_path)
        if folders=="test":
            for files in testing_labels:
                existing_file_path=os.path.join("data","labels",files)
                if os.path.exists(existing_file_path):
                    new_file_path = os.path.join("data",folders,"labels",files)
                    os.mkdir(new_file_path)
                    os.replace(existing_file_path,new_file_path)
            for files in testing_images:
                existing_file_paths = os.path.join("data","images",files)
                if os.path.exists(existing_file_path):
                    new_file_path = os.path.join("data",folders,"images",files)
                    os.mkdir(new_file_path)
                    os.replace(existing_file_paths,new_file_path)
# Not calling the move_partition_data() function due to incoming errors

# Preparing the data augmentation pipeline (diversifying data by 30x)
# Defining an augmentor object that shall act as the main diversification pipeline-

augmentor = alb.Compose([
    alb.RandomCrop(width=450,height=450),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),
    alb.HorizontalFlip(p=0.5),
    alb.VerticalFlip(p=0.5),
    alb.RandomFog(fog_coef_lower=0.3,fog_coef_upper=1,alpha_coef=0.05,always_apply=False,p=0.5),
    alb.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20,drop_width=1, drop_color=(200, 200, 200), blur_value=7,brightness_coefficient=0.7, rain_type=None, always_apply=False, p=0.5),
    alb.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1,num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=0.5)],
    bbox_params=alb.BboxParams(format='albumentations',label_fields=['class_labels'])) 
# Testing this augmentation pipeline on one of our images->
testing_img = cv2.imread(os.path.join("data","images","71a97e4f-f927-11ec-9cc9-dce99469a2b8.jpg"))
print("This is the image selected for testing->")
print(testing_img)

with open(os.path.join("data","labels","71a97e4f-f927-11ec-9cc9-dce99469a2b8.json"),'r') as f:
    testing_label = json.load(f) #Loading a corresponding label for our test_image, basically to obtain the coordinates of the bbox.
print(testing_label)
# Extracting the coordinates of the bounding boxes:
coords=[0,0,0,0]
coords[0] = testing_label['shapes'][0]['points'][0][0]
coords[1] = testing_label['shapes'][0]['points'][0][1]
coords[2] = testing_label['shapes'][0]['points'][1][0]
coords[3] = testing_label['shapes'][0]['points'][1][1]
print(f"The onbtained coordinates are- {coords}")
# Normalizing the obtained coordinates in order to fit our format of 'albumentaions' in the bbox_params in the data augmentation pipeline
coords = list(np.divide(coords,[640,480,640,480]))
print(f"The normalized coordinates are as follows-{coords}")
# Checking the augmented images as a dictionary->
augmented = augmentor(image = testing_img,bboxes=[coords],class_labels=['face'])
print(f"The augmented object obtained from the augmentor is a {type(augmented)}")
print(augmented)
# Visualizing our augmented image->
cv2.rectangle(
    augmented['image'],
    tuple(np.multiply(augmented['bboxes'][0][:2],[450,450]).astype(int)),
    tuple(np.multiply(augmented['bboxes'][0][2:],[450,450]).astype(int)),
    (255,0,0),2
)
print("Showing an image from data with applied augmentation")
plt.imshow(augmented['image'])
plt.show()
# Defining a proper augmentation pipeline with matching labels and images->
trigger_augmentatiion = input("Do you want to generate augmented images and labels?").lower()
trigger_values = ['y','yes','true','ok']
stop_values = ['no','n','false']
while trigger_augmentatiion in trigger_values or trigger_augmentatiion in stop_values:
    if trigger_augmentatiion in trigger_values:
        print("Starting augmentation")
        for image in os.listdir(os.path.join('data','images')):
            img = cv2.imread(os.path.join('data','images', image))

            coords = [0,0,0.00001,0.00001]
            label_path = os.path.join('data','labels', f'{image.split(".")[0]}.json')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label = json.load(f)

                coords[0] = label['shapes'][0]['points'][0][0]
                coords[1] = label['shapes'][0]['points'][0][1]
                coords[2] = label['shapes'][0]['points'][1][0]
                coords[3] = label['shapes'][0]['points'][1][1]
                coords = list(np.divide(coords, [640,480,640,480]))

            try:
                for x in range(60):
                    augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                    cv2.imwrite(os.path.join("data","augmented_images", f'{image.split(".")[0]}.{x}.jpg'),augmented['image'])

                    annotation = {}
                    annotation['image'] = image

                    if os.path.exists(label_path):
                        if len(augmented['bboxes']) == 0:
                            annotation['bbox'] = [0,0,0,0]
                            annotation['class'] = 0
                        else:
                            annotation['bbox'] = augmented['bboxes'][0]
                            annotation['class'] = 1
                    else:
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0


                    with open(os.path.join("data","augmented_labels" f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                        json.dump(annotation, f)

            except Exception as e:
                print(e)
        break
    elif trigger_augmentatiion in stop_values:
        break
    else:
        print("please enter a valid value")
        trigger_augmentatiion = input("Please enter a valid value")

for items in os.listdir("data"):
    for x in range(60):
        if items==f"{items.split('.')[0]}.{x}.json":
            os.replace(os.path.join("data",items),os.path.join("data","augmented_labels",items))

trigger_partition=input("Do you want the images to be partitioned?\n")

all_aug_img_test = []
all_aug_img_train = []
all_aug_label_test = []
all_aug_label_train = []

for items in os.listdir(path = os.path.join("data","test","images")): #Seperating the testing images
    print(items)
    if items not in all_aug_img_test:
        all_aug_img_test.append(items)
    print("Item appended to testing images list")
for items in os.listdir(path = os.path.join("data","train","images")):
    if items not in all_aug_img_train:
     all_aug_img_train.append(items)
    print(f"{items} appended to training images list")

print(len(all_aug_img_test))
print(len(all_aug_img_train))

trigger_label_partition = input("Do you want to partition the labels?").lower()
if trigger_label_partition=="yes":
    for items in all_aug_img_test:
        # print(items) #Line only for testing purposes
        file_name = f"augmented_labels{items.split('.')[0]}.{items.split('.')[1]}.json"
        # print(f"The file currently being processed is - {file_name}") #Line only  for testing purposes
        # # writing a sub_for loop for matching obtained image names with their corresponding labels
        for files in os.listdir(path = os.path.join("data","augmented_labels")):
            if file_name==files:
                # print("yes") #This line was for testing purposes only
                if not os.path.exists(os.path.join("data","test","labels",files)):
                    os.replace(os.path.join("data","augmented_labels",files),os.path.join("data","test","labels",files))
                    print(f"{files:5}-shifted")
                else:
                    print(f"{files:5} already exists")
    for items in all_aug_img_train:
        file_name = f"augmented_labels{items.split('.')[0]}.{items.split('.')[1]}.json"
        for files in os.listdir(os.path.join("data","augmented_labels")):
            if file_name==files:
                if not os.path.exists(os.path.join("data","train","labels",files)):
                    os.replace(os.path.join("data","augmented_labels",files),os.path.join("data","train","labels",files))
                    print(f"{files:5} shifted for training")
                else:
                    print(f"{files:5} already in target lcation")
else:
    pass
# Loadiing the uagmented images to tensorflow dataset->
train_images = tf.data.Dataset.list_files("data\\train\\images\\*.jpg", shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x,(120,120)))
train_images = train_images.map(lambda x: x/255)
# Doing the same for testing images->
test_images = tf.data.Dataset.list_files("data\\test\\images\\*.jpg", shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x,(120,120)))
test_images = test_images.map(lambda x: x/255)
# Doing the same for validation images->
val_images = tf.data.Dataset.list_files("data\\images\\*.jpg", shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x,(120,120)))
val_images = val_images.map(lambda x: x/255)

print([len(train_images),len(test_images),len(val_images)])
print([len(train_images)/8,len(test_images)/8,len(val_images)/8])
# Making a function to load in the labels:
def load_label(label_path):
    with open(label_path.numpy(), 'r' , encoding='utf-8') as f:
        label = json.load(f)
        return [label['class'],label['bbox']]

# Loading our labels to the tensorflow dataset->
train_labels = tf.data.Dataset.list_files("data\\train\\labels\\*.json",shuffle=False)
func_map = lambda x: tf.py_function(load_Label,[x],[tf.uint8,tf.float16])
train_labels = train_labels.map(func_map)
# Testing labels->
test_label = tf.data.Dataset.list_files("data\\test\\labels\\*.json",shuffle=False)
test_label = test_label.map(func_map)
# Validation labels->
val_labels = tf.data.Dataset.list_files("data\\labels\\*.json",shuffle=False)
val_labels = val_labels.map(func_map)

# print(train_labels.as_numpy_iterator().next()) #Line meant for testing purposes
# Checking the length of our images andd labels
print("PRINTING LENGTHS OF IMAGES AND LABELS")
print([len(train_images),len(train_labels)])
print([len(test_images),len(test_label)])
print([len(val_images),len(val_labels)])

# PREPARING THE FINAL DATASET TO BE FED INTO THE MODEL
train = tf.data.Dataset.zip((train_images,train_labels))
train = train.shuffle(1600)
train = train.batch(8,drop_remainder=True)
train = train.prefetch(4)
# Testing data
test = tf.data.Dataset.zip((test_images,test_label))
test = test.shuffle(800)
test = test.batch(8,drop_remainder=True)
test = test.prefetch(4)
# Validation data
val = tf.data.Dataset.zip((val_images,val_labels))
val = val.shuffle(400)
val = val.batch(8,drop_remainder=True)
val = val.prefetch(4)

print(type(train))

print(test.as_numpy_iterator().next()[0].shape) #Num, dimension and color layers of our images
print(test.as_numpy_iterator().next()[1]) #Label annotation

# Viewing the augmented and annotated images
data_sample = test.as_numpy_iterator()
res = data_sample.next()
# Plotting the image
fig,ax = plt.subplots(ncols=4,figsize=(20,20))
for idx in range(4):
    sample_image = res[0][idx]
    sample_coords = res[1][1][idx]
    cv2.rectangle(
        sample_image,
        tuple(np.multiply(sample_coords[:2],[120,120]).astype(int)),
        tuple(np.multiply(sample_coords[2:],[120,120]).astype(int)),
        (255,0,0),2
    )

# Loading in VGG16
vgg = VGG16(include_top = False)
# Looking at the details of our VGG16 NeuralNet
vgg.summary()
# Building the NN_>
def build_model():
    input_layer = Input(shape=(120,120,3))
    # This is to define the classifications output
    vgg = VGG16(include_top=False)(input_layer)
    f1  = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048,activation="relu")(f1)
    class2 = Dense(1, activation = "sigmoid")(class1)

    # This is to define the regression output for the value points
    f2 = GlobalMaxPooling2D()(vgg)
    rgeress1 = Dense(2048, activation="relu")(f2)
    regress2 = Dense(4, activation = "sigmoid")(rgeress1)
    facetracker = Model(inputs = input_layer, outputs = [class2,regress2]) #This shall track the faces in the camera input

    return facetracker

print(test.as_numpy_iterator().next()[1])
# Building the face tracker NN->
facetracker  = build_model()
print("Here is the final facetracker Neural Network")
facetracker.summary()
# Dry testing it with making arbitrary predictions->
test_NN = input("Do you want to test the facetracker NN?\n").lower()
if test_NN=="yes":
    X,y = train.as_numpy_iterator().next()
    classes, coords = facetracker.predict(X)
    print(classes)
    print(coords)
else:
    pass
# Defining custom loss and optimizer functions->
batches_per_epoch = len(train)
learning_rate_decay = (1./0.75-1)/batches_per_epoch
print(f"learning_rate_decay {learning_rate_decay:5}")
"""
The learning rate decay is to reduce the learning rate while keeping it at 75% of the actual rate
This is mainly to avoid over-fitting and making it go out of range
This shall reduce it by a constant rate per epoch (75%)
"""
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, decay = learning_rate_decay)
# Creating localization loss mainly for regression and classification
def localization_loss(y_true, yhat):
    yhat = tf.cast(yhat,tf.float32)
    y_true = tf.cast(y_true,tf.float32)
    delta_coords = tf.reduce_sum(tf.square(y_true[:,:2]-yhat[:,:2])) # Obtaining the height and width through y-axis and squaring for x-axis
    height_true = y_true[:,:3]- yhat[:,:1] #Getting true height
    width_true = y_true[:2] - yhat[:,0] #Getting true width

    height_predicted = y_true[:,:3] - yhat[:,:1] #Obtaining the predicted height
    width_predicted = y_true[:2] - yhat[:,0] #Obtaining the predicted height

    delta_size = tf.reduce_sum(tf.square(width_true-width_predicted)+tf.square(height_true-height_predicted)) #Basically obtaining the distance b/w actual and predicted coordinates
    #Returning all these calculations as the sum; which shall be our loss.
    return delta_coords+delta_size

class_loss = tf.keras.losses.BinaryCrossentropy()
regress_loss = localization_loss

print(class_loss) #Just for inspecting purposes out of curiosity


X, y = train.as_numpy_iterator().next()

def testing_losses(probabilities, coords, classes):
    print("Here is the overall localization loss")
    print(localization_loss(probabilities[1],coords)) #-> To get the actual values
    print("The classification loss->")
    print(class_loss(probabilities[0],classes))
    print("The regression loss")
    print(regress_loss(probabilities[1],coords))

test_loss = input("Do you wish to test the losses?\n").lower()
if test_loss=="yes":
    testing_losses(probabilities=y, coords=coords,classes=classes)
else:
    pass

# Building a training pipeline-
# Defining a ccustom model class :

class FaceTracker(Model):
    # Making the initialization function to instantiate our model fro training
    def __init__(self,facetracker,**kwargs):
        super().__init__(**kwargs)
        self.model = facetracker
    # Making the compile method to compile all the losses and optimizers needed to train the model
    def compile(self, optimizer,classloss,localization_loss,**kwargs):
        super().compile(**kwargs)
        self.classloss = class_loss
        self.localization_loss = regress_loss
        self.optimizer = optimizer
    # Defining the training step where the backprop shall be applied to our data
    def train_step(self,batch,**kwargs):

        X,y = batch
        with tf.GradientTape() as tape:
            classes,coords = self.model(X,training=True)
            batch_classloss = self.classloss(y[0],classes)
            batch_localloss = self.localization_loss(y[1],coords)

            total_loss = batch_localloss+0.5*batch_classloss
            grad = tape.gradient(total_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grad,self.model.trainable_variables))
            return{
                "total_loss":total_loss,
                "class_loss":batch_classloss,
                "localization_loss":batch_localloss
            }
        # Definging the testing step for the model, which is similar to the train_step but without backprop
        def test_step(self,batch,**kwargs):
            X,y = batch
            classes,coords = self.model(X,training=True)
            batch_classloss = self.classloss(y[0],classes)
            batch_localloss = self.localization_loss(tf.cast(y[1],tf.float32),coords)
            total_loss = batch_localloss+0.5%batch_classloss
            return{
                "total_loss":total_loss,
                "classloss":batch_classloss,
                "localization_loss":batch_localloss
            }
        #Defining a lambda function that should act as a call function.Lambda is used so as to prevent us from having to subclass the method to make a child class
        lambda self,X,**kwargs: self.model(X,**kwargs)

# Subclassing our facetracker model for training
model = FaceTracker(facetracker)
model.compile(optimizer,localization_loss,class_loss)
logdir = "logdir"
tensorflow_callbacks = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join("callbacks"),
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only= True
)
model.fit(train,epochs=10,validation_data=val,callbacks=[tensorflow_callbacks])

