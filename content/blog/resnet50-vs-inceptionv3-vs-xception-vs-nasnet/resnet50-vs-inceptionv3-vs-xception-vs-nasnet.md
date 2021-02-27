---
title: "ResNet50 vs InceptionV3 vs Xception vs NASNet - Introduction to Transfer Learning"
description: "Transfer learning is an ML methodology that enables to reuse a model developed for one task to another task. The applications are predominantly in Deep Learning for computer vision and natural language processing."
lead: "Transfer learning is an ML methodology that enables to reuse a model developed for one task to another task. The applications are predominantly in Deep Learning for computer vision and natural language processing."
date: 2020-11-04T09:19:42+01:00
lastmod: 2020-11-04T09:19:42+01:00
draft: false
weight: 10
images: []
contributors: ["Gowri Shankar"]
image: "https://miro.medium.com/max/1000/1*jQZ_oJ3VZbVWiBWuc5SMMg.png"
---

Objective:

Objective of this kernel is to introduce transfer learning to beginners. I have taken the following deep neural network applications 
- ResNet50
- InceptionV3
- Xception
- NASNet

**Accuracy versus Computational Demand (Left) and Number of Parameters (Right)**
![Accuracy versus Computational Demand (Left) and Number of Parameters (Right)
](https://miro.medium.com/max/1000/1*jQZ_oJ3VZbVWiBWuc5SMMg.png)

### Transfer Learning
Transfer learning is an ML methodology that enables to reuse a model developed for one task to another task.
The applications are predominantly in Deep Learning for computer vision and natural language processing.
This kernel introduces one on how to use Keras transfer learning applications.


## ResNet50 (APTOS Accuracy: 0.396)
Created By: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
Literature: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)  
Topological Depth: **152 Layers**  
ImageNet Validation Accuracy: **Top-1 Accuracy**: 0.749 **Top-5 Accuracy**: 0.921  

## InceptionV3 (APTOS Accuracy: 0.559)
Created By: Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna  
Literature: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)  
Topological Depth: **159 Layers**  
ImageNet Validation Accuracy: **Top-1 Accuracy**: 0.779 **Top-5 Accuracy**: 0.937 

## Xception (APTOS Accuracy: 0.509)
Created By: François Chollet  
Literature: [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)  
Topological Depth: **126 Layers**  
ImageNet Validation Accuracy: **Top-1 Accuracy**: 0.790 **Top-5 Accuracy**: 0.945 

## NASNet (APTOS Accuracy: TBD)
Created By: Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le  
Literature: [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)  
Topological Depth: **~1040**  
ImageNet Validation Accuracy: **Top-1 Accuracy**: 0.825 **Top-5 Accuracy**: 0.960 



```python
import os
print(os.listdir("../input"))

```

    ['aptos2019-blindness-detection', 'keras-pretrained-models', 'nasnetlarge']


## Preprocess


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
print("Shape of train data: {0}".format(train_df.shape))
test_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
print("Shape of test data: {0}".format(test_df.shape))

diagnosis_df = pd.DataFrame({
    'diagnosis': [0, 1, 2, 3, 4],
    'diagnosis_label': ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
})

train_df = train_df.merge(diagnosis_df, how="left", on="diagnosis")

train_image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("../input/aptos2019-blindness-detection/train_images")) for f in fn]
train_images_df = pd.DataFrame({
    'files': train_image_files,
    'id_code': [file.split('/')[4].split('.')[0] for file in train_image_files],
})
train_df = train_df.merge(train_images_df, how="left", on="id_code")
del train_images_df
print("Shape of train data: {0}".format(train_df.shape))

test_image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("../input/aptos2019-blindness-detection/test_images")) for f in fn]
test_images_df = pd.DataFrame({
    'files': test_image_files,
    'id_code': [file.split('/')[4].split('.')[0] for file in test_image_files],
})


test_df = test_df.merge(test_images_df, how="left", on="id_code")
del test_images_df
print("Shape of test data: {0}".format(test_df.shape))

# Any results you write to the current directory are saved as output.
```

    Shape of train data: (3662, 2)
    Shape of test data: (1928, 1)
    Shape of train data: (3662, 4)
    Shape of test data: (1928, 2)



```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_code</th>
      <th>diagnosis</th>
      <th>diagnosis_label</th>
      <th>files</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000c1434d8d7</td>
      <td>2</td>
      <td>Moderate</td>
      <td>../input/aptos2019-blindness-detection/train_i...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>001639a390f0</td>
      <td>4</td>
      <td>Proliferative DR</td>
      <td>../input/aptos2019-blindness-detection/train_i...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0024cdab0c1e</td>
      <td>1</td>
      <td>Mild</td>
      <td>../input/aptos2019-blindness-detection/train_i...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>002c21358ce6</td>
      <td>0</td>
      <td>No DR</td>
      <td>../input/aptos2019-blindness-detection/train_i...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>005b95c28852</td>
      <td>0</td>
      <td>No DR</td>
      <td>../input/aptos2019-blindness-detection/train_i...</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_code</th>
      <th>files</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0005cfc8afb6</td>
      <td>../input/aptos2019-blindness-detection/test_im...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>003f0afdcd15</td>
      <td>../input/aptos2019-blindness-detection/test_im...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>006efc72b638</td>
      <td>../input/aptos2019-blindness-detection/test_im...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00836aaacf06</td>
      <td>../input/aptos2019-blindness-detection/test_im...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>009245722fa4</td>
      <td>../input/aptos2019-blindness-detection/test_im...</td>
    </tr>
  </tbody>
</table>
</div>




```python
IMG_SIZE = 150
N_CLASSES = train_df.diagnosis.nunique()
CLASSES = list(map(str, range(N_CLASSES)))
BATCH_SIZE = 32
EPOCH_STEPS = 10
EPOCHS = 25
NB_FILTERS = 32
KERNEL_SIZE = 4
CHANNELS = 3
```

## Data Generator: Train, Validation and Test Datasets


```python
import tensorflow as tf
print(tf.__version__)

from keras.preprocessing.image import ImageDataGenerator

train_df["diagnosis"] = train_df["diagnosis"].astype(str)

train_data_gen = ImageDataGenerator(
    # featurewise_center = True, # Set input mean to 0 over the dataset
    samplewise_center = True, # set each sample mean to 0
    featurewise_std_normalization = True, # Divide inputs by std of the dataset
    samplewise_std_normalization = True, # Divide each input by its std
    # zca_whitening = True, # Apply ZCA whitening
    zca_epsilon = 1e-06, # Epsilon for ZCA whitening,
    rotation_range = 30, # randomly rotate imges in the range (degrees, 0 to 189)
    width_shift_range = 0.1, # randomly shift images horizontally (fraction of total width)
    height_shift_range = 0.1, # Randomly shift images vertically (fraction of total height)
    shear_range = 0, # set range for random shear
    zoom_range = [0.75, 1.25], # set range for random zoom
    channel_shift_range = 0.05, # set range for random channel shifts
    fill_mode = 'constant', # set mode for filling points outside the input boundaries
    cval = 0, # value used for fill_mode
    horizontal_flip = True,
    vertical_flip = True,
    rescale = 1/255.,
    preprocessing_function = None,
    validation_split=0.1
)
train_data = train_data_gen.flow_from_dataframe(
    dataframe=train_df, 
    x_col="files",
    y_col="diagnosis",
    batch_size=BATCH_SIZE,
    shuffle=True,
    classes=CLASSES,
    class_mode="categorical",
    target_size=(IMG_SIZE, IMG_SIZE),
    subset="training"
)

validation_data = train_data_gen.flow_from_dataframe(
    dataframe=train_df, 
    x_col="files",
    y_col="diagnosis",
    batch_size=BATCH_SIZE,
    shuffle=True,
    classes=CLASSES,
    class_mode="categorical",
    target_size=(IMG_SIZE, IMG_SIZE),
    subset="validation"
)

test_data_gen = ImageDataGenerator(rescale=1./255)
test_data = test_data_gen.flow_from_dataframe(
    dataframe=test_df,
    x_col="files",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size = 1,
    shuffle=False,
    class_mode=None
)
```

## Transfer Learning Assets


```python
from tensorflow.python.keras.applications import ResNet50, InceptionV3, Xception, NASNetLarge
print(os.listdir(("../input/keras-pretrained-models/")))
print(os.listdir(("../input/nasnetlarge/")))

model_resnet50 = ResNet50(
    weights="../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", 
    include_top=False, 
    input_tensor=tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
)


model_inception_v3 = InceptionV3(
    weights="../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5", 
    include_top=False, 
    input_tensor=tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
)

model_xception = Xception(
    weights="../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5", 
    include_top=False, 
    input_tensor=tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
)

model_nasnet_large = NASNetLarge(
    weights="../input/nasnetlarge/NASNet-large-no-top.h5", 
    include_top=False, 
    input_tensor=tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
)

```

    ['inception_v3_weights_tf_dim_ordering_tf_kernels.h5', 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', 'imagenet_class_index.json', 'xception_weights_tf_dim_ordering_tf_kernels_notop.h5', 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5', 'xception_weights_tf_dim_ordering_tf_kernels.h5', 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5', 'resnet50_weights_tf_dim_ordering_tf_kernels.h5', 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'Kuszma.JPG']
    ['NASNet-large-no-top.h5']


    /opt/conda/lib/python3.6/site-packages/keras_applications/resnet50.py:265: UserWarning: The output shape of `ResNet50(include_top=False)` has been changed since Keras 2.2.0.
      warnings.warn('The output shape of `ResNet50(include_top=False)` '



```python


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(NB_FILTERS, (KERNEL_SIZE, KERNEL_SIZE), padding="valid", strides=1, input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS), activation="relu"),
        tf.keras.layers.Conv2D(NB_FILTERS, (KERNEL_SIZE, KERNEL_SIZE), activation="relu"),
        tf.keras.layers.Conv2D(NB_FILTERS, (KERNEL_SIZE, KERNEL_SIZE), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(8, 8)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(2048, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(N_CLASSES, activation="softmax")
        
    ])
    return model
```


```python
# Resnet50: 0.396
# create_model: 0.152
# InceptionV3: 0.559
# Xception: 0.509

def get_model(model):
    X = model.output

    X = tf.keras.layers.GlobalAveragePooling2D()(X)
    X = tf.keras.layers.Dense(2048, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.25)(X)
    X = tf.keras.layers.Dense(1024, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.25)(X)
    X = tf.keras.layers.Dense(512, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.25)(X)
    X = tf.keras.layers.Dense(256, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.25)(X)
    X = tf.keras.layers.Dense(128, activation='relu')(X)
    predictions = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(X)
    model = tf.keras.Model(inputs=model.input, outputs=predictions)
    
#     for layer in model.layers:
#         layer.trainable = True
        
#     for layer in model.layers[15:]:
#         layer.trainable = False
    
    return model


```

## Optimize, Compile, Train and Predict


```python
optimizer=tf.keras.optimizers.Nadam(lr=2*1e-3, schedule_decay=1e-5)
```


```python
algo = "inception_v3"
klass = "basics"

model = get_model(model_inception_v3)

opt = tf.keras.optimizers.Adam(lr=0.001, epsilon=1e-6)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


```

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_2 (InputLayer)            [(None, 150, 150, 3) 0                                            
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, 74, 74, 32)   864         input_2[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization (BatchNorma (None, 74, 74, 32)   96          conv2d[0][0]                     
    __________________________________________________________________________________________________
    activation_49 (Activation)      (None, 74, 74, 32)   0           batch_normalization[0][0]        
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 72, 72, 32)   9216        activation_49[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, 72, 72, 32)   96          conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    activation_50 (Activation)      (None, 72, 72, 32)   0           batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 72, 72, 64)   18432       activation_50[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, 72, 72, 64)   192         conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    activation_51 (Activation)      (None, 72, 72, 64)   0           batch_normalization_2[0][0]      
    __________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)  (None, 35, 35, 64)   0           activation_51[0][0]              
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, 35, 35, 80)   5120        max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_3 (BatchNor (None, 35, 35, 80)   240         conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    activation_52 (Activation)      (None, 35, 35, 80)   0           batch_normalization_3[0][0]      
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, 33, 33, 192)  138240      activation_52[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_4 (BatchNor (None, 33, 33, 192)  576         conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    activation_53 (Activation)      (None, 33, 33, 192)  0           batch_normalization_4[0][0]      
    __________________________________________________________________________________________________
    max_pooling2d_2 (MaxPooling2D)  (None, 16, 16, 192)  0           activation_53[0][0]              
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, 16, 16, 64)   12288       max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_8 (BatchNor (None, 16, 16, 64)   192         conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    activation_57 (Activation)      (None, 16, 16, 64)   0           batch_normalization_8[0][0]      
    __________________________________________________________________________________________________
    conv2d_6 (Conv2D)               (None, 16, 16, 48)   9216        max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    conv2d_9 (Conv2D)               (None, 16, 16, 96)   55296       activation_57[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_6 (BatchNor (None, 16, 16, 48)   144         conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_9 (BatchNor (None, 16, 16, 96)   288         conv2d_9[0][0]                   
    __________________________________________________________________________________________________
    activation_55 (Activation)      (None, 16, 16, 48)   0           batch_normalization_6[0][0]      
    __________________________________________________________________________________________________
    activation_58 (Activation)      (None, 16, 16, 96)   0           batch_normalization_9[0][0]      
    __________________________________________________________________________________________________
    average_pooling2d (AveragePooli (None, 16, 16, 192)  0           max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, 16, 16, 64)   12288       max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    conv2d_7 (Conv2D)               (None, 16, 16, 64)   76800       activation_55[0][0]              
    __________________________________________________________________________________________________
    conv2d_10 (Conv2D)              (None, 16, 16, 96)   82944       activation_58[0][0]              
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, 16, 16, 32)   6144        average_pooling2d[0][0]          
    __________________________________________________________________________________________________
    batch_normalization_5 (BatchNor (None, 16, 16, 64)   192         conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_7 (BatchNor (None, 16, 16, 64)   192         conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_10 (BatchNo (None, 16, 16, 96)   288         conv2d_10[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_11 (BatchNo (None, 16, 16, 32)   96          conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    activation_54 (Activation)      (None, 16, 16, 64)   0           batch_normalization_5[0][0]      
    __________________________________________________________________________________________________
    activation_56 (Activation)      (None, 16, 16, 64)   0           batch_normalization_7[0][0]      
    __________________________________________________________________________________________________
    activation_59 (Activation)      (None, 16, 16, 96)   0           batch_normalization_10[0][0]     
    __________________________________________________________________________________________________
    activation_60 (Activation)      (None, 16, 16, 32)   0           batch_normalization_11[0][0]     
    __________________________________________________________________________________________________
    mixed0 (Concatenate)            (None, 16, 16, 256)  0           activation_54[0][0]              
                                                                     activation_56[0][0]              
                                                                     activation_59[0][0]              
                                                                     activation_60[0][0]              
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, 16, 16, 64)   16384       mixed0[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_15 (BatchNo (None, 16, 16, 64)   192         conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    activation_64 (Activation)      (None, 16, 16, 64)   0           batch_normalization_15[0][0]     
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, 16, 16, 48)   12288       mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, 16, 16, 96)   55296       activation_64[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_13 (BatchNo (None, 16, 16, 48)   144         conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_16 (BatchNo (None, 16, 16, 96)   288         conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    activation_62 (Activation)      (None, 16, 16, 48)   0           batch_normalization_13[0][0]     
    __________________________________________________________________________________________________
    activation_65 (Activation)      (None, 16, 16, 96)   0           batch_normalization_16[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_1 (AveragePoo (None, 16, 16, 256)  0           mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, 16, 16, 64)   16384       mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, 16, 16, 64)   76800       activation_62[0][0]              
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, 16, 16, 96)   82944       activation_65[0][0]              
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, 16, 16, 64)   16384       average_pooling2d_1[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_12 (BatchNo (None, 16, 16, 64)   192         conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_14 (BatchNo (None, 16, 16, 64)   192         conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_17 (BatchNo (None, 16, 16, 96)   288         conv2d_17[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_18 (BatchNo (None, 16, 16, 64)   192         conv2d_18[0][0]                  
    __________________________________________________________________________________________________
    activation_61 (Activation)      (None, 16, 16, 64)   0           batch_normalization_12[0][0]     
    __________________________________________________________________________________________________
    activation_63 (Activation)      (None, 16, 16, 64)   0           batch_normalization_14[0][0]     
    __________________________________________________________________________________________________
    activation_66 (Activation)      (None, 16, 16, 96)   0           batch_normalization_17[0][0]     
    __________________________________________________________________________________________________
    activation_67 (Activation)      (None, 16, 16, 64)   0           batch_normalization_18[0][0]     
    __________________________________________________________________________________________________
    mixed1 (Concatenate)            (None, 16, 16, 288)  0           activation_61[0][0]              
                                                                     activation_63[0][0]              
                                                                     activation_66[0][0]              
                                                                     activation_67[0][0]              
    __________________________________________________________________________________________________
    conv2d_22 (Conv2D)              (None, 16, 16, 64)   18432       mixed1[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_22 (BatchNo (None, 16, 16, 64)   192         conv2d_22[0][0]                  
    __________________________________________________________________________________________________
    activation_71 (Activation)      (None, 16, 16, 64)   0           batch_normalization_22[0][0]     
    __________________________________________________________________________________________________
    conv2d_20 (Conv2D)              (None, 16, 16, 48)   13824       mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_23 (Conv2D)              (None, 16, 16, 96)   55296       activation_71[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_20 (BatchNo (None, 16, 16, 48)   144         conv2d_20[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_23 (BatchNo (None, 16, 16, 96)   288         conv2d_23[0][0]                  
    __________________________________________________________________________________________________
    activation_69 (Activation)      (None, 16, 16, 48)   0           batch_normalization_20[0][0]     
    __________________________________________________________________________________________________
    activation_72 (Activation)      (None, 16, 16, 96)   0           batch_normalization_23[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_2 (AveragePoo (None, 16, 16, 288)  0           mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_19 (Conv2D)              (None, 16, 16, 64)   18432       mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_21 (Conv2D)              (None, 16, 16, 64)   76800       activation_69[0][0]              
    __________________________________________________________________________________________________
    conv2d_24 (Conv2D)              (None, 16, 16, 96)   82944       activation_72[0][0]              
    __________________________________________________________________________________________________
    conv2d_25 (Conv2D)              (None, 16, 16, 64)   18432       average_pooling2d_2[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_19 (BatchNo (None, 16, 16, 64)   192         conv2d_19[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_21 (BatchNo (None, 16, 16, 64)   192         conv2d_21[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_24 (BatchNo (None, 16, 16, 96)   288         conv2d_24[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_25 (BatchNo (None, 16, 16, 64)   192         conv2d_25[0][0]                  
    __________________________________________________________________________________________________
    activation_68 (Activation)      (None, 16, 16, 64)   0           batch_normalization_19[0][0]     
    __________________________________________________________________________________________________
    activation_70 (Activation)      (None, 16, 16, 64)   0           batch_normalization_21[0][0]     
    __________________________________________________________________________________________________
    activation_73 (Activation)      (None, 16, 16, 96)   0           batch_normalization_24[0][0]     
    __________________________________________________________________________________________________
    activation_74 (Activation)      (None, 16, 16, 64)   0           batch_normalization_25[0][0]     
    __________________________________________________________________________________________________
    mixed2 (Concatenate)            (None, 16, 16, 288)  0           activation_68[0][0]              
                                                                     activation_70[0][0]              
                                                                     activation_73[0][0]              
                                                                     activation_74[0][0]              
    __________________________________________________________________________________________________
    conv2d_27 (Conv2D)              (None, 16, 16, 64)   18432       mixed2[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_27 (BatchNo (None, 16, 16, 64)   192         conv2d_27[0][0]                  
    __________________________________________________________________________________________________
    activation_76 (Activation)      (None, 16, 16, 64)   0           batch_normalization_27[0][0]     
    __________________________________________________________________________________________________
    conv2d_28 (Conv2D)              (None, 16, 16, 96)   55296       activation_76[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_28 (BatchNo (None, 16, 16, 96)   288         conv2d_28[0][0]                  
    __________________________________________________________________________________________________
    activation_77 (Activation)      (None, 16, 16, 96)   0           batch_normalization_28[0][0]     
    __________________________________________________________________________________________________
    conv2d_26 (Conv2D)              (None, 7, 7, 384)    995328      mixed2[0][0]                     
    __________________________________________________________________________________________________
    conv2d_29 (Conv2D)              (None, 7, 7, 96)     82944       activation_77[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_26 (BatchNo (None, 7, 7, 384)    1152        conv2d_26[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_29 (BatchNo (None, 7, 7, 96)     288         conv2d_29[0][0]                  
    __________________________________________________________________________________________________
    activation_75 (Activation)      (None, 7, 7, 384)    0           batch_normalization_26[0][0]     
    __________________________________________________________________________________________________
    activation_78 (Activation)      (None, 7, 7, 96)     0           batch_normalization_29[0][0]     
    __________________________________________________________________________________________________
    max_pooling2d_3 (MaxPooling2D)  (None, 7, 7, 288)    0           mixed2[0][0]                     
    __________________________________________________________________________________________________
    mixed3 (Concatenate)            (None, 7, 7, 768)    0           activation_75[0][0]              
                                                                     activation_78[0][0]              
                                                                     max_pooling2d_3[0][0]            
    __________________________________________________________________________________________________
    conv2d_34 (Conv2D)              (None, 7, 7, 128)    98304       mixed3[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_34 (BatchNo (None, 7, 7, 128)    384         conv2d_34[0][0]                  
    __________________________________________________________________________________________________
    activation_83 (Activation)      (None, 7, 7, 128)    0           batch_normalization_34[0][0]     
    __________________________________________________________________________________________________
    conv2d_35 (Conv2D)              (None, 7, 7, 128)    114688      activation_83[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_35 (BatchNo (None, 7, 7, 128)    384         conv2d_35[0][0]                  
    __________________________________________________________________________________________________
    activation_84 (Activation)      (None, 7, 7, 128)    0           batch_normalization_35[0][0]     
    __________________________________________________________________________________________________
    conv2d_31 (Conv2D)              (None, 7, 7, 128)    98304       mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_36 (Conv2D)              (None, 7, 7, 128)    114688      activation_84[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_31 (BatchNo (None, 7, 7, 128)    384         conv2d_31[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_36 (BatchNo (None, 7, 7, 128)    384         conv2d_36[0][0]                  
    __________________________________________________________________________________________________
    activation_80 (Activation)      (None, 7, 7, 128)    0           batch_normalization_31[0][0]     
    __________________________________________________________________________________________________
    activation_85 (Activation)      (None, 7, 7, 128)    0           batch_normalization_36[0][0]     
    __________________________________________________________________________________________________
    conv2d_32 (Conv2D)              (None, 7, 7, 128)    114688      activation_80[0][0]              
    __________________________________________________________________________________________________
    conv2d_37 (Conv2D)              (None, 7, 7, 128)    114688      activation_85[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_32 (BatchNo (None, 7, 7, 128)    384         conv2d_32[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_37 (BatchNo (None, 7, 7, 128)    384         conv2d_37[0][0]                  
    __________________________________________________________________________________________________
    activation_81 (Activation)      (None, 7, 7, 128)    0           batch_normalization_32[0][0]     
    __________________________________________________________________________________________________
    activation_86 (Activation)      (None, 7, 7, 128)    0           batch_normalization_37[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_3 (AveragePoo (None, 7, 7, 768)    0           mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_30 (Conv2D)              (None, 7, 7, 192)    147456      mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_33 (Conv2D)              (None, 7, 7, 192)    172032      activation_81[0][0]              
    __________________________________________________________________________________________________
    conv2d_38 (Conv2D)              (None, 7, 7, 192)    172032      activation_86[0][0]              
    __________________________________________________________________________________________________
    conv2d_39 (Conv2D)              (None, 7, 7, 192)    147456      average_pooling2d_3[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_30 (BatchNo (None, 7, 7, 192)    576         conv2d_30[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_33 (BatchNo (None, 7, 7, 192)    576         conv2d_33[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_38 (BatchNo (None, 7, 7, 192)    576         conv2d_38[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_39 (BatchNo (None, 7, 7, 192)    576         conv2d_39[0][0]                  
    __________________________________________________________________________________________________
    activation_79 (Activation)      (None, 7, 7, 192)    0           batch_normalization_30[0][0]     
    __________________________________________________________________________________________________
    activation_82 (Activation)      (None, 7, 7, 192)    0           batch_normalization_33[0][0]     
    __________________________________________________________________________________________________
    activation_87 (Activation)      (None, 7, 7, 192)    0           batch_normalization_38[0][0]     
    __________________________________________________________________________________________________
    activation_88 (Activation)      (None, 7, 7, 192)    0           batch_normalization_39[0][0]     
    __________________________________________________________________________________________________
    mixed4 (Concatenate)            (None, 7, 7, 768)    0           activation_79[0][0]              
                                                                     activation_82[0][0]              
                                                                     activation_87[0][0]              
                                                                     activation_88[0][0]              
    __________________________________________________________________________________________________
    conv2d_44 (Conv2D)              (None, 7, 7, 160)    122880      mixed4[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_44 (BatchNo (None, 7, 7, 160)    480         conv2d_44[0][0]                  
    __________________________________________________________________________________________________
    activation_93 (Activation)      (None, 7, 7, 160)    0           batch_normalization_44[0][0]     
    __________________________________________________________________________________________________
    conv2d_45 (Conv2D)              (None, 7, 7, 160)    179200      activation_93[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_45 (BatchNo (None, 7, 7, 160)    480         conv2d_45[0][0]                  
    __________________________________________________________________________________________________
    activation_94 (Activation)      (None, 7, 7, 160)    0           batch_normalization_45[0][0]     
    __________________________________________________________________________________________________
    conv2d_41 (Conv2D)              (None, 7, 7, 160)    122880      mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_46 (Conv2D)              (None, 7, 7, 160)    179200      activation_94[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_41 (BatchNo (None, 7, 7, 160)    480         conv2d_41[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_46 (BatchNo (None, 7, 7, 160)    480         conv2d_46[0][0]                  
    __________________________________________________________________________________________________
    activation_90 (Activation)      (None, 7, 7, 160)    0           batch_normalization_41[0][0]     
    __________________________________________________________________________________________________
    activation_95 (Activation)      (None, 7, 7, 160)    0           batch_normalization_46[0][0]     
    __________________________________________________________________________________________________
    conv2d_42 (Conv2D)              (None, 7, 7, 160)    179200      activation_90[0][0]              
    __________________________________________________________________________________________________
    conv2d_47 (Conv2D)              (None, 7, 7, 160)    179200      activation_95[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_42 (BatchNo (None, 7, 7, 160)    480         conv2d_42[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_47 (BatchNo (None, 7, 7, 160)    480         conv2d_47[0][0]                  
    __________________________________________________________________________________________________
    activation_91 (Activation)      (None, 7, 7, 160)    0           batch_normalization_42[0][0]     
    __________________________________________________________________________________________________
    activation_96 (Activation)      (None, 7, 7, 160)    0           batch_normalization_47[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_4 (AveragePoo (None, 7, 7, 768)    0           mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_40 (Conv2D)              (None, 7, 7, 192)    147456      mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_43 (Conv2D)              (None, 7, 7, 192)    215040      activation_91[0][0]              
    __________________________________________________________________________________________________
    conv2d_48 (Conv2D)              (None, 7, 7, 192)    215040      activation_96[0][0]              
    __________________________________________________________________________________________________
    conv2d_49 (Conv2D)              (None, 7, 7, 192)    147456      average_pooling2d_4[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_40 (BatchNo (None, 7, 7, 192)    576         conv2d_40[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_43 (BatchNo (None, 7, 7, 192)    576         conv2d_43[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_48 (BatchNo (None, 7, 7, 192)    576         conv2d_48[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_49 (BatchNo (None, 7, 7, 192)    576         conv2d_49[0][0]                  
    __________________________________________________________________________________________________
    activation_89 (Activation)      (None, 7, 7, 192)    0           batch_normalization_40[0][0]     
    __________________________________________________________________________________________________
    activation_92 (Activation)      (None, 7, 7, 192)    0           batch_normalization_43[0][0]     
    __________________________________________________________________________________________________
    activation_97 (Activation)      (None, 7, 7, 192)    0           batch_normalization_48[0][0]     
    __________________________________________________________________________________________________
    activation_98 (Activation)      (None, 7, 7, 192)    0           batch_normalization_49[0][0]     
    __________________________________________________________________________________________________
    mixed5 (Concatenate)            (None, 7, 7, 768)    0           activation_89[0][0]              
                                                                     activation_92[0][0]              
                                                                     activation_97[0][0]              
                                                                     activation_98[0][0]              
    __________________________________________________________________________________________________
    conv2d_54 (Conv2D)              (None, 7, 7, 160)    122880      mixed5[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_54 (BatchNo (None, 7, 7, 160)    480         conv2d_54[0][0]                  
    __________________________________________________________________________________________________
    activation_103 (Activation)     (None, 7, 7, 160)    0           batch_normalization_54[0][0]     
    __________________________________________________________________________________________________
    conv2d_55 (Conv2D)              (None, 7, 7, 160)    179200      activation_103[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_55 (BatchNo (None, 7, 7, 160)    480         conv2d_55[0][0]                  
    __________________________________________________________________________________________________
    activation_104 (Activation)     (None, 7, 7, 160)    0           batch_normalization_55[0][0]     
    __________________________________________________________________________________________________
    conv2d_51 (Conv2D)              (None, 7, 7, 160)    122880      mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_56 (Conv2D)              (None, 7, 7, 160)    179200      activation_104[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_51 (BatchNo (None, 7, 7, 160)    480         conv2d_51[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_56 (BatchNo (None, 7, 7, 160)    480         conv2d_56[0][0]                  
    __________________________________________________________________________________________________
    activation_100 (Activation)     (None, 7, 7, 160)    0           batch_normalization_51[0][0]     
    __________________________________________________________________________________________________
    activation_105 (Activation)     (None, 7, 7, 160)    0           batch_normalization_56[0][0]     
    __________________________________________________________________________________________________
    conv2d_52 (Conv2D)              (None, 7, 7, 160)    179200      activation_100[0][0]             
    __________________________________________________________________________________________________
    conv2d_57 (Conv2D)              (None, 7, 7, 160)    179200      activation_105[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_52 (BatchNo (None, 7, 7, 160)    480         conv2d_52[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_57 (BatchNo (None, 7, 7, 160)    480         conv2d_57[0][0]                  
    __________________________________________________________________________________________________
    activation_101 (Activation)     (None, 7, 7, 160)    0           batch_normalization_52[0][0]     
    __________________________________________________________________________________________________
    activation_106 (Activation)     (None, 7, 7, 160)    0           batch_normalization_57[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_5 (AveragePoo (None, 7, 7, 768)    0           mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_50 (Conv2D)              (None, 7, 7, 192)    147456      mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_53 (Conv2D)              (None, 7, 7, 192)    215040      activation_101[0][0]             
    __________________________________________________________________________________________________
    conv2d_58 (Conv2D)              (None, 7, 7, 192)    215040      activation_106[0][0]             
    __________________________________________________________________________________________________
    conv2d_59 (Conv2D)              (None, 7, 7, 192)    147456      average_pooling2d_5[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_50 (BatchNo (None, 7, 7, 192)    576         conv2d_50[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_53 (BatchNo (None, 7, 7, 192)    576         conv2d_53[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_58 (BatchNo (None, 7, 7, 192)    576         conv2d_58[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_59 (BatchNo (None, 7, 7, 192)    576         conv2d_59[0][0]                  
    __________________________________________________________________________________________________
    activation_99 (Activation)      (None, 7, 7, 192)    0           batch_normalization_50[0][0]     
    __________________________________________________________________________________________________
    activation_102 (Activation)     (None, 7, 7, 192)    0           batch_normalization_53[0][0]     
    __________________________________________________________________________________________________
    activation_107 (Activation)     (None, 7, 7, 192)    0           batch_normalization_58[0][0]     
    __________________________________________________________________________________________________
    activation_108 (Activation)     (None, 7, 7, 192)    0           batch_normalization_59[0][0]     
    __________________________________________________________________________________________________
    mixed6 (Concatenate)            (None, 7, 7, 768)    0           activation_99[0][0]              
                                                                     activation_102[0][0]             
                                                                     activation_107[0][0]             
                                                                     activation_108[0][0]             
    __________________________________________________________________________________________________
    conv2d_64 (Conv2D)              (None, 7, 7, 192)    147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_64 (BatchNo (None, 7, 7, 192)    576         conv2d_64[0][0]                  
    __________________________________________________________________________________________________
    activation_113 (Activation)     (None, 7, 7, 192)    0           batch_normalization_64[0][0]     
    __________________________________________________________________________________________________
    conv2d_65 (Conv2D)              (None, 7, 7, 192)    258048      activation_113[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_65 (BatchNo (None, 7, 7, 192)    576         conv2d_65[0][0]                  
    __________________________________________________________________________________________________
    activation_114 (Activation)     (None, 7, 7, 192)    0           batch_normalization_65[0][0]     
    __________________________________________________________________________________________________
    conv2d_61 (Conv2D)              (None, 7, 7, 192)    147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_66 (Conv2D)              (None, 7, 7, 192)    258048      activation_114[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_61 (BatchNo (None, 7, 7, 192)    576         conv2d_61[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_66 (BatchNo (None, 7, 7, 192)    576         conv2d_66[0][0]                  
    __________________________________________________________________________________________________
    activation_110 (Activation)     (None, 7, 7, 192)    0           batch_normalization_61[0][0]     
    __________________________________________________________________________________________________
    activation_115 (Activation)     (None, 7, 7, 192)    0           batch_normalization_66[0][0]     
    __________________________________________________________________________________________________
    conv2d_62 (Conv2D)              (None, 7, 7, 192)    258048      activation_110[0][0]             
    __________________________________________________________________________________________________
    conv2d_67 (Conv2D)              (None, 7, 7, 192)    258048      activation_115[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_62 (BatchNo (None, 7, 7, 192)    576         conv2d_62[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_67 (BatchNo (None, 7, 7, 192)    576         conv2d_67[0][0]                  
    __________________________________________________________________________________________________
    activation_111 (Activation)     (None, 7, 7, 192)    0           batch_normalization_62[0][0]     
    __________________________________________________________________________________________________
    activation_116 (Activation)     (None, 7, 7, 192)    0           batch_normalization_67[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_6 (AveragePoo (None, 7, 7, 768)    0           mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_60 (Conv2D)              (None, 7, 7, 192)    147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_63 (Conv2D)              (None, 7, 7, 192)    258048      activation_111[0][0]             
    __________________________________________________________________________________________________
    conv2d_68 (Conv2D)              (None, 7, 7, 192)    258048      activation_116[0][0]             
    __________________________________________________________________________________________________
    conv2d_69 (Conv2D)              (None, 7, 7, 192)    147456      average_pooling2d_6[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_60 (BatchNo (None, 7, 7, 192)    576         conv2d_60[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_63 (BatchNo (None, 7, 7, 192)    576         conv2d_63[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_68 (BatchNo (None, 7, 7, 192)    576         conv2d_68[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_69 (BatchNo (None, 7, 7, 192)    576         conv2d_69[0][0]                  
    __________________________________________________________________________________________________
    activation_109 (Activation)     (None, 7, 7, 192)    0           batch_normalization_60[0][0]     
    __________________________________________________________________________________________________
    activation_112 (Activation)     (None, 7, 7, 192)    0           batch_normalization_63[0][0]     
    __________________________________________________________________________________________________
    activation_117 (Activation)     (None, 7, 7, 192)    0           batch_normalization_68[0][0]     
    __________________________________________________________________________________________________
    activation_118 (Activation)     (None, 7, 7, 192)    0           batch_normalization_69[0][0]     
    __________________________________________________________________________________________________
    mixed7 (Concatenate)            (None, 7, 7, 768)    0           activation_109[0][0]             
                                                                     activation_112[0][0]             
                                                                     activation_117[0][0]             
                                                                     activation_118[0][0]             
    __________________________________________________________________________________________________
    conv2d_72 (Conv2D)              (None, 7, 7, 192)    147456      mixed7[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_72 (BatchNo (None, 7, 7, 192)    576         conv2d_72[0][0]                  
    __________________________________________________________________________________________________
    activation_121 (Activation)     (None, 7, 7, 192)    0           batch_normalization_72[0][0]     
    __________________________________________________________________________________________________
    conv2d_73 (Conv2D)              (None, 7, 7, 192)    258048      activation_121[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_73 (BatchNo (None, 7, 7, 192)    576         conv2d_73[0][0]                  
    __________________________________________________________________________________________________
    activation_122 (Activation)     (None, 7, 7, 192)    0           batch_normalization_73[0][0]     
    __________________________________________________________________________________________________
    conv2d_70 (Conv2D)              (None, 7, 7, 192)    147456      mixed7[0][0]                     
    __________________________________________________________________________________________________
    conv2d_74 (Conv2D)              (None, 7, 7, 192)    258048      activation_122[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_70 (BatchNo (None, 7, 7, 192)    576         conv2d_70[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_74 (BatchNo (None, 7, 7, 192)    576         conv2d_74[0][0]                  
    __________________________________________________________________________________________________
    activation_119 (Activation)     (None, 7, 7, 192)    0           batch_normalization_70[0][0]     
    __________________________________________________________________________________________________
    activation_123 (Activation)     (None, 7, 7, 192)    0           batch_normalization_74[0][0]     
    __________________________________________________________________________________________________
    conv2d_71 (Conv2D)              (None, 3, 3, 320)    552960      activation_119[0][0]             
    __________________________________________________________________________________________________
    conv2d_75 (Conv2D)              (None, 3, 3, 192)    331776      activation_123[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_71 (BatchNo (None, 3, 3, 320)    960         conv2d_71[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_75 (BatchNo (None, 3, 3, 192)    576         conv2d_75[0][0]                  
    __________________________________________________________________________________________________
    activation_120 (Activation)     (None, 3, 3, 320)    0           batch_normalization_71[0][0]     
    __________________________________________________________________________________________________
    activation_124 (Activation)     (None, 3, 3, 192)    0           batch_normalization_75[0][0]     
    __________________________________________________________________________________________________
    max_pooling2d_4 (MaxPooling2D)  (None, 3, 3, 768)    0           mixed7[0][0]                     
    __________________________________________________________________________________________________
    mixed8 (Concatenate)            (None, 3, 3, 1280)   0           activation_120[0][0]             
                                                                     activation_124[0][0]             
                                                                     max_pooling2d_4[0][0]            
    __________________________________________________________________________________________________
    conv2d_80 (Conv2D)              (None, 3, 3, 448)    573440      mixed8[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_80 (BatchNo (None, 3, 3, 448)    1344        conv2d_80[0][0]                  
    __________________________________________________________________________________________________
    activation_129 (Activation)     (None, 3, 3, 448)    0           batch_normalization_80[0][0]     
    __________________________________________________________________________________________________
    conv2d_77 (Conv2D)              (None, 3, 3, 384)    491520      mixed8[0][0]                     
    __________________________________________________________________________________________________
    conv2d_81 (Conv2D)              (None, 3, 3, 384)    1548288     activation_129[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_77 (BatchNo (None, 3, 3, 384)    1152        conv2d_77[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_81 (BatchNo (None, 3, 3, 384)    1152        conv2d_81[0][0]                  
    __________________________________________________________________________________________________
    activation_126 (Activation)     (None, 3, 3, 384)    0           batch_normalization_77[0][0]     
    __________________________________________________________________________________________________
    activation_130 (Activation)     (None, 3, 3, 384)    0           batch_normalization_81[0][0]     
    __________________________________________________________________________________________________
    conv2d_78 (Conv2D)              (None, 3, 3, 384)    442368      activation_126[0][0]             
    __________________________________________________________________________________________________
    conv2d_79 (Conv2D)              (None, 3, 3, 384)    442368      activation_126[0][0]             
    __________________________________________________________________________________________________
    conv2d_82 (Conv2D)              (None, 3, 3, 384)    442368      activation_130[0][0]             
    __________________________________________________________________________________________________
    conv2d_83 (Conv2D)              (None, 3, 3, 384)    442368      activation_130[0][0]             
    __________________________________________________________________________________________________
    average_pooling2d_7 (AveragePoo (None, 3, 3, 1280)   0           mixed8[0][0]                     
    __________________________________________________________________________________________________
    conv2d_76 (Conv2D)              (None, 3, 3, 320)    409600      mixed8[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_78 (BatchNo (None, 3, 3, 384)    1152        conv2d_78[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_79 (BatchNo (None, 3, 3, 384)    1152        conv2d_79[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_82 (BatchNo (None, 3, 3, 384)    1152        conv2d_82[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_83 (BatchNo (None, 3, 3, 384)    1152        conv2d_83[0][0]                  
    __________________________________________________________________________________________________
    conv2d_84 (Conv2D)              (None, 3, 3, 192)    245760      average_pooling2d_7[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_76 (BatchNo (None, 3, 3, 320)    960         conv2d_76[0][0]                  
    __________________________________________________________________________________________________
    activation_127 (Activation)     (None, 3, 3, 384)    0           batch_normalization_78[0][0]     
    __________________________________________________________________________________________________
    activation_128 (Activation)     (None, 3, 3, 384)    0           batch_normalization_79[0][0]     
    __________________________________________________________________________________________________
    activation_131 (Activation)     (None, 3, 3, 384)    0           batch_normalization_82[0][0]     
    __________________________________________________________________________________________________
    activation_132 (Activation)     (None, 3, 3, 384)    0           batch_normalization_83[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_84 (BatchNo (None, 3, 3, 192)    576         conv2d_84[0][0]                  
    __________________________________________________________________________________________________
    activation_125 (Activation)     (None, 3, 3, 320)    0           batch_normalization_76[0][0]     
    __________________________________________________________________________________________________
    mixed9_0 (Concatenate)          (None, 3, 3, 768)    0           activation_127[0][0]             
                                                                     activation_128[0][0]             
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 3, 3, 768)    0           activation_131[0][0]             
                                                                     activation_132[0][0]             
    __________________________________________________________________________________________________
    activation_133 (Activation)     (None, 3, 3, 192)    0           batch_normalization_84[0][0]     
    __________________________________________________________________________________________________
    mixed9 (Concatenate)            (None, 3, 3, 2048)   0           activation_125[0][0]             
                                                                     mixed9_0[0][0]                   
                                                                     concatenate[0][0]                
                                                                     activation_133[0][0]             
    __________________________________________________________________________________________________
    conv2d_89 (Conv2D)              (None, 3, 3, 448)    917504      mixed9[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_89 (BatchNo (None, 3, 3, 448)    1344        conv2d_89[0][0]                  
    __________________________________________________________________________________________________
    activation_138 (Activation)     (None, 3, 3, 448)    0           batch_normalization_89[0][0]     
    __________________________________________________________________________________________________
    conv2d_86 (Conv2D)              (None, 3, 3, 384)    786432      mixed9[0][0]                     
    __________________________________________________________________________________________________
    conv2d_90 (Conv2D)              (None, 3, 3, 384)    1548288     activation_138[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_86 (BatchNo (None, 3, 3, 384)    1152        conv2d_86[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_90 (BatchNo (None, 3, 3, 384)    1152        conv2d_90[0][0]                  
    __________________________________________________________________________________________________
    activation_135 (Activation)     (None, 3, 3, 384)    0           batch_normalization_86[0][0]     
    __________________________________________________________________________________________________
    activation_139 (Activation)     (None, 3, 3, 384)    0           batch_normalization_90[0][0]     
    __________________________________________________________________________________________________
    conv2d_87 (Conv2D)              (None, 3, 3, 384)    442368      activation_135[0][0]             
    __________________________________________________________________________________________________
    conv2d_88 (Conv2D)              (None, 3, 3, 384)    442368      activation_135[0][0]             
    __________________________________________________________________________________________________
    conv2d_91 (Conv2D)              (None, 3, 3, 384)    442368      activation_139[0][0]             
    __________________________________________________________________________________________________
    conv2d_92 (Conv2D)              (None, 3, 3, 384)    442368      activation_139[0][0]             
    __________________________________________________________________________________________________
    average_pooling2d_8 (AveragePoo (None, 3, 3, 2048)   0           mixed9[0][0]                     
    __________________________________________________________________________________________________
    conv2d_85 (Conv2D)              (None, 3, 3, 320)    655360      mixed9[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_87 (BatchNo (None, 3, 3, 384)    1152        conv2d_87[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_88 (BatchNo (None, 3, 3, 384)    1152        conv2d_88[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_91 (BatchNo (None, 3, 3, 384)    1152        conv2d_91[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_92 (BatchNo (None, 3, 3, 384)    1152        conv2d_92[0][0]                  
    __________________________________________________________________________________________________
    conv2d_93 (Conv2D)              (None, 3, 3, 192)    393216      average_pooling2d_8[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_85 (BatchNo (None, 3, 3, 320)    960         conv2d_85[0][0]                  
    __________________________________________________________________________________________________
    activation_136 (Activation)     (None, 3, 3, 384)    0           batch_normalization_87[0][0]     
    __________________________________________________________________________________________________
    activation_137 (Activation)     (None, 3, 3, 384)    0           batch_normalization_88[0][0]     
    __________________________________________________________________________________________________
    activation_140 (Activation)     (None, 3, 3, 384)    0           batch_normalization_91[0][0]     
    __________________________________________________________________________________________________
    activation_141 (Activation)     (None, 3, 3, 384)    0           batch_normalization_92[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_93 (BatchNo (None, 3, 3, 192)    576         conv2d_93[0][0]                  
    __________________________________________________________________________________________________
    activation_134 (Activation)     (None, 3, 3, 320)    0           batch_normalization_85[0][0]     
    __________________________________________________________________________________________________
    mixed9_1 (Concatenate)          (None, 3, 3, 768)    0           activation_136[0][0]             
                                                                     activation_137[0][0]             
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 3, 3, 768)    0           activation_140[0][0]             
                                                                     activation_141[0][0]             
    __________________________________________________________________________________________________
    activation_142 (Activation)     (None, 3, 3, 192)    0           batch_normalization_93[0][0]     
    __________________________________________________________________________________________________
    mixed10 (Concatenate)           (None, 3, 3, 2048)   0           activation_134[0][0]             
                                                                     mixed9_1[0][0]                   
                                                                     concatenate_1[0][0]              
                                                                     activation_142[0][0]             
    __________________________________________________________________________________________________
    global_average_pooling2d (Globa (None, 2048)         0           mixed10[0][0]                    
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 2048)         4196352     global_average_pooling2d[0][0]   
    __________________________________________________________________________________________________
    dropout (Dropout)               (None, 2048)         0           dense[0][0]                      
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 1024)         2098176     dropout[0][0]                    
    __________________________________________________________________________________________________
    dropout_1 (Dropout)             (None, 1024)         0           dense_1[0][0]                    
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 512)          524800      dropout_1[0][0]                  
    __________________________________________________________________________________________________
    dropout_2 (Dropout)             (None, 512)          0           dense_2[0][0]                    
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 256)          131328      dropout_2[0][0]                  
    __________________________________________________________________________________________________
    dropout_3 (Dropout)             (None, 256)          0           dense_3[0][0]                    
    __________________________________________________________________________________________________
    dense_4 (Dense)                 (None, 128)          32896       dropout_3[0][0]                  
    __________________________________________________________________________________________________
    dense_5 (Dense)                 (None, 5)            645         dense_4[0][0]                    
    ==================================================================================================
    Total params: 28,786,981
    Trainable params: 28,752,549
    Non-trainable params: 34,432
    __________________________________________________________________________________________________



```python
from sklearn.metrics import cohen_kappa_score
class QWKCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.X = validation_data[0]
        self.Y = validation_data[1]
        self.history = []
        
    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.X)
        score = cohen_kappa_score(
            np.argmax(self.Y, axis=1), np.argmax(pred, axis=1), labels=[0, 1, 2, 3, 4], weights="quadratic"
        )
        print(("Epoch {0} : QWK : {1}".format(epoch, score)))
        self.history.append(score)
        if(score >= max(self.history)):
            print("Saving Checkpoint: {0}".format(score))
            self.model.save("../Resnet50_bestQWK.h5")
```


```python
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", 
    min_delta=0.0001, patience=3, verbose=1, mode="auto")
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", 
    min_delta=0.0004, patience=2, factor=0.1, min_lr=1e-6, mode="auto", verbose=1)
```


```python
qwk = QWKCallback(validation_data)
model.fit_generator(
    generator=train_data,
    #steps_per_epochs=EPOCH_STEPS,
    #batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_data,
    validation_steps=30#,
    #callbacks=[early_stopping, reduce_lr]
)
```

    /opt/conda/lib/python3.6/site-packages/keras_preprocessing/image/image_data_generator.py:716: UserWarning: This ImageDataGenerator specifies `featurewise_center`, but it hasn't been fit on any training data. Fit it first by calling `.fit(numpy_data)`.
      warnings.warn('This ImageDataGenerator specifies '
    /opt/conda/lib/python3.6/site-packages/keras_preprocessing/image/image_data_generator.py:724: UserWarning: This ImageDataGenerator specifies `featurewise_std_normalization`, but it hasn't been fit on any training data. Fit it first by calling `.fit(numpy_data)`.
      warnings.warn('This ImageDataGenerator specifies '


    Epoch 1/25


    /opt/conda/lib/python3.6/site-packages/keras_preprocessing/image/image_data_generator.py:716: UserWarning: This ImageDataGenerator specifies `featurewise_center`, but it hasn't been fit on any training data. Fit it first by calling `.fit(numpy_data)`.
      warnings.warn('This ImageDataGenerator specifies '
    /opt/conda/lib/python3.6/site-packages/keras_preprocessing/image/image_data_generator.py:724: UserWarning: This ImageDataGenerator specifies `featurewise_std_normalization`, but it hasn't been fit on any training data. Fit it first by calling `.fit(numpy_data)`.
      warnings.warn('This ImageDataGenerator specifies '


    103/103 [==============================] - 503s 5s/step - loss: 1.2704 - acc: 0.5507 - val_loss: 3530.0762 - val_acc: 0.4816
    Epoch 2/25
    103/103 [==============================] - 448s 4s/step - loss: 1.0078 - acc: 0.6748 - val_loss: 21.8471 - val_acc: 0.4946
    Epoch 3/25
    103/103 [==============================] - 440s 4s/step - loss: 0.9933 - acc: 0.6663 - val_loss: 87.6621 - val_acc: 0.4903
    Epoch 4/25
    103/103 [==============================] - 443s 4s/step - loss: 0.9498 - acc: 0.6878 - val_loss: 190.1645 - val_acc: 0.5444
    Epoch 5/25
    103/103 [==============================] - 445s 4s/step - loss: 0.8962 - acc: 0.6878 - val_loss: 1022.9081 - val_acc: 0.1374
    Epoch 6/25
    103/103 [==============================] - 441s 4s/step - loss: 0.9300 - acc: 0.6854 - val_loss: 965.7076 - val_acc: 0.5584
    Epoch 7/25
    103/103 [==============================] - 448s 4s/step - loss: 0.9197 - acc: 0.6893 - val_loss: 14.3867 - val_acc: 0.4968
    Epoch 8/25
    103/103 [==============================] - 444s 4s/step - loss: 0.8321 - acc: 0.7127 - val_loss: 1.2525 - val_acc: 0.7100
    Epoch 9/25
    103/103 [==============================] - 448s 4s/step - loss: 0.8426 - acc: 0.7042 - val_loss: 1.7017 - val_acc: 0.6115
    Epoch 10/25
    103/103 [==============================] - 445s 4s/step - loss: 0.8222 - acc: 0.7130 - val_loss: 8.9885 - val_acc: 0.6450
    Epoch 11/25
    103/103 [==============================] - 437s 4s/step - loss: 0.8559 - acc: 0.7133 - val_loss: 202.3459 - val_acc: 0.5032
    Epoch 12/25
    103/103 [==============================] - 441s 4s/step - loss: 0.8583 - acc: 0.7118 - val_loss: 1.1618 - val_acc: 0.6645
    Epoch 13/25
    103/103 [==============================] - 447s 4s/step - loss: 0.8247 - acc: 0.7166 - val_loss: 1.2152 - val_acc: 0.7132
    Epoch 14/25
    103/103 [==============================] - 446s 4s/step - loss: 1.0001 - acc: 0.6602 - val_loss: 830.5764 - val_acc: 0.4329
    Epoch 15/25
    103/103 [==============================] - 447s 4s/step - loss: 0.9909 - acc: 0.6602 - val_loss: 43.9164 - val_acc: 0.5216
    Epoch 16/25
    103/103 [==============================] - 446s 4s/step - loss: 0.8771 - acc: 0.6969 - val_loss: 1.4656 - val_acc: 0.6418
    Epoch 17/25
    103/103 [==============================] - 448s 4s/step - loss: 0.8122 - acc: 0.7175 - val_loss: 1.0396 - val_acc: 0.6948
    Epoch 18/25
    103/103 [==============================] - 441s 4s/step - loss: 0.8221 - acc: 0.7118 - val_loss: 0.8343 - val_acc: 0.7284
    Epoch 19/25
    103/103 [==============================] - 447s 4s/step - loss: 0.7953 - acc: 0.7148 - val_loss: 1.1735 - val_acc: 0.6721
    Epoch 20/25
    103/103 [==============================] - 444s 4s/step - loss: 0.8761 - acc: 0.6990 - val_loss: 4.6178 - val_acc: 0.3831
    Epoch 21/25
    103/103 [==============================] - 442s 4s/step - loss: 0.8527 - acc: 0.7042 - val_loss: 1.2654 - val_acc: 0.6331
    Epoch 22/25
    103/103 [==============================] - 445s 4s/step - loss: 0.8787 - acc: 0.6984 - val_loss: 5.4739 - val_acc: 0.6699
    Epoch 23/25
    103/103 [==============================] - 439s 4s/step - loss: 0.8281 - acc: 0.7148 - val_loss: 0.8836 - val_acc: 0.6894
    Epoch 24/25
    103/103 [==============================] - 438s 4s/step - loss: 0.7883 - acc: 0.7212 - val_loss: 0.7943 - val_acc: 0.7045
    Epoch 25/25
    103/103 [==============================] - 438s 4s/step - loss: 0.8209 - acc: 0.7093 - val_loss: 1.5930 - val_acc: 0.6061





    <tensorflow.python.keras.callbacks.History at 0x7fb2dac07c50>




```python
filenames = test_data.filenames
classifications = model.predict_generator(test_data, steps=len(filenames))
```


```python
results = pd.DataFrame({
    "id_code": filenames,
    "diagnosis": np.argmax(classifications, axis=1)
})
results["id_code"] = results["id_code"].map(lambda x: str(x)[:-4].split("/")[4])
results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_code</th>
      <th>diagnosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0005cfc8afb6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>003f0afdcd15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>006efc72b638</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00836aaacf06</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>009245722fa4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
file_name = "{0}_{1}.csv".format(algo, klass)
results.to_csv("submission.csv", index=False)
```


```python
results.diagnosis.value_counts()

len(model.layers)
```




    322




```python

```
