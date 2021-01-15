import os
from keras import layers, optimizers, models
from keras.applications.resnet import ResNet152, preprocess_input
from keras.layers import *    
from keras.models import Model
train_dir = os.path.join(r'D:\jupyterlab\training')
validation_dir = os.path.join(r'D:\jupyterlab\val')
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
Reduce=ReduceLROnPlateau(
    monitor ='val_loss',#监测的值，可以是accuracy，val_loss,val_accuracy
    factor=0.1,#缩放学习率的值，学习率将以lr = lr*factor的形式被减少
    patience=2,#当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    verbose=1,
    mode='auto',#‘auto’，‘min’，‘max’之一 默认‘auto’就行
    cooldown=0,#学习率减少后，会经过cooldown个epoch才重新进行正常操作
    min_lr=0 #学习率最小值，能缩小到的下限
)
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=10
)
resnet152 = ResNet152(weights='imagenet', include_top=False, input_shape=(300,300, 3))

model = models.Sequential()
model.add(resnet152)
model.add(GlobalAveragePooling2D())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
# model.add(layers.Flatten())
# model.add(Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

resnet152.trainable = False
#冻结一个层意味着将其排除在训练之外，即其权重将永远不会更新
optimizer = optimizers.RMSprop(lr=1e-4)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
lr_metric = get_lr_metric(optimizer)
model.compile(loss='binary_crossentropy', optimizer = optimizers.RMSprop(lr=1e-4), metrics=['acc',lr_metric])

from keras.preprocessing.image import ImageDataGenerator

batch_size = 64

train_datagen = ImageDataGenerator(
    rotation_range=45,#随机旋转
    width_shift_range=0.2,#是图像在水平上平移的范围
    height_shift_range=0.2,#垂直方向上平移的范围
    shear_range=0.2,#随机错切变换的角度
    zoom_range=0.2,#随机缩放的范围
    horizontal_flip=True,#随机将图像水平翻转
    preprocessing_function=preprocess_input#归一化
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(300,300),
        batch_size=batch_size,
        class_mode='binary')
validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(300,300),
        batch_size=batch_size,
        class_mode='binary')

model.save('cat-dog-resnet152-epoch30.h5')