from keras.models import Sequential
from keras.layers import *
from keras import applications as kapps
from keras.optimizers import RMSprop, Adam
from keras.models import Model


def getNetByName(model_name, n_classes=6):
    if model_name == 'SilNet':
        model = SilNet(n_classes)
        image_size = 150
    elif  model_name == 'WeiNet':
        model = WeiNet(n_classes)
        image_size = 64
    elif  model_name == 'AlexNet':
        model = AlexNet(n_classes)
        image_size = 224
    elif  model_name == 'SongNet':
        model = SongNet(n_classes)
        image_size = 224
    else:
        image_size = 224
        input_layer = Input(shape=(image_size, image_size, 3))
        if  model_name == 'InceptionV3':
            base_model = kapps.InceptionV3(weights='imagenet', include_top=False, input_tensor=input_layer)
        elif  model_name == 'VGG19':
            base_model = kapps.VGG19(weights='imagenet', include_top=False, input_tensor=input_layer)
        elif  model_name == 'VGG16':
            base_model = kapps.VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)
        elif  model_name == 'ResNet50':
            base_model = kapps.ResNet50(weights='imagenet', include_top=False, input_tensor=input_layer)
        elif  model_name == 'ResNet101V2':
            base_model = kapps.ResNet101V2(weights='imagenet', include_top=False, input_tensor=input_layer)
        elif  model_name == 'Xception':
            base_model = kapps.Xception(weights='imagenet', include_top=False, input_tensor=input_layer)
        elif  model_name == 'MobileNetV3Large':
            base_model = kapps.MobileNetV3Large(weights='imagenet', include_top=False, input_tensor=input_layer)
        elif  model_name == 'EfficientNetV2B0':
            base_model = kapps.EfficientNetV2B0(weights='imagenet', include_top=False, input_tensor=input_layer)
        elif  model_name == 'ConvNeXtTiny':
            base_model = kapps.ConvNeXtTiny(weights='imagenet', include_top=False, input_tensor=input_layer)

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(n_classes, activation='softmax')(x)

        model = Model(input_layer, x)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])
    
    return model, image_size


def getNetByNamePain(model_name, n_classes=6):
    if model_name == 'SilNet':
        model = SilNet(n_classes)
        image_size = 150
    elif  model_name == 'WeiNet':
        model = WeiNet(n_classes)
        image_size = 64
    elif  model_name == 'AlexNet':
        model = AlexNet(n_classes)
        image_size = 224
    elif  model_name == 'SongNet':
        model = SongNet(n_classes)
        image_size = 224
    else:
        image_size = 224
        model = Sequential()
        model.add(Input(shape=(image_size, image_size, 3)))
        if  model_name == 'InceptionV3':
            model.add(kapps.InceptionV3(weights='imagenet', include_top=False))
        elif  model_name == 'VGG19':
            model.add(kapps.VGG19(weights='imagenet', include_top=False))
        elif  model_name == 'VGG16':
            model.add(kapps.VGG16(weights='imagenet', include_top=False))
        elif  model_name == 'ResNet50':
            model.add(kapps.ResNet50(weights='imagenet', include_top=False))
        elif  model_name == 'ResNet101V2':
            model.add(kapps.ResNet101V2(weights='imagenet', include_top=False))
        elif  model_name == 'Xception':
            model.add(kapps.Xception(weights='imagenet', include_top=False))
        elif  model_name == 'MobileNetV3Large':
            model.add(kapps.MobileNetV3Large(weights='imagenet', include_top=False))
        elif  model_name == 'EfficientNetV2B0':
            model.add(kapps.EfficientNetV2B0(weights='imagenet', include_top=False))
        elif  model_name == 'ConvNeXtTiny':
            model.add(kapps.ConvNeXtTiny(weights='imagenet', include_top=False))

        model.add(GlobalAveragePooling2D())
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

    return model, image_size


def AlexNet(n_classes=6):
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))

    model.add(Conv2D(96, kernel_size = 11, strides= (4, 4), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same'))

    model.add(Conv2D(256, kernel_size = 5, strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))

    model.add(Conv2D(384, kernel_size = 3, strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))

    model.add(Conv2D(384, kernel_size = 3, strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))

    model.add(Conv2D(256, kernel_size = 3, strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))

    model.add(Flatten())
    model.add(Dense(4096, input_shape=(224*224*3,), activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])

    return model


def WeiNet(n_classes=6):
    model = Sequential()
    model.add(Input(shape=(64, 64, 3)))

    model.add(Conv2D(32, (7, 7), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    model.add(Conv2D(32, (7, 7), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    model.add(Conv2D(64, (7, 7), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])
    return model


def SongNet(n_classes=6):
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))

    model.add(Conv2D(64, (5, 5), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Conv2D(64, (5, 5), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Conv2D(32, (3, 3), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])
    return model


def SilNet(n_classes=6):
    model = Sequential()
    model.add(Input(shape=(150, 150, 3)))

    model.add(Conv2D(32, (11, 11), padding='same', data_format='channels_last', kernel_initializer='glorot_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(32, (7, 7), padding='same', data_format='channels_last', kernel_initializer='glorot_normal', activation='relu'))

    model.add(Conv2D(32, (7, 7), padding='same', data_format='channels_last', kernel_initializer='glorot_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    model.add(Conv2D(64, (5, 5), padding='same', data_format='channels_last', kernel_initializer='glorot_normal', activation='relu'))
    
    model.add(Conv2D(64, (5, 5), padding='same', data_format='channels_last', kernel_initializer='glorot_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    model.add(Flatten())
    model.add(Dense(512, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    return model
