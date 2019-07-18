import numpy as np
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import train_test_split
import keras
from keras.layers import *
from keras.models import Model
from keras.optimizers import *
from keras.regularizers import *
from keras.callbacks import Callback
from latent_hyper_net import LatentHyperNet

class LearningRateScheduler(Callback):

    def __init__(self, init_lr=0.01, schedule=[(25, 1e-2), (50, 1e-3), (100, 1e-4)]):
        super(Callback, self).__init__()
        self.init_lr = init_lr
        self.schedule = schedule

    def on_epoch_end(self, epoch, logs={}):
        lr = self.init_lr
        for i in range(0, len(self.schedule) - 1):
            if epoch >= self.schedule[i][0] and epoch < self.schedule[i + 1][0]:
                lr = self.schedule[i][1]

        if epoch >= self.schedule[-1][0]:
            lr = self.schedule[-1][1]

        print('Learning rate:{}'.format(lr))
        #K.set_value(self.model.optimizer.lr, lr)
        keras.backend.set_value(self.model.optimizer.lr, lr)

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_model(architecture_file='', weights_file=''):
    if '.json' not in architecture_file:
        architecture_file = architecture_file+'.json'

    with open(architecture_file, 'r') as f:
        model = keras.models.model_from_json(f.read())

    if weights_file != '':
        if '.h5' not in weights_file:
            weights_file = weights_file + '.h5'
        model.load_weights(weights_file)
        print('Load architecture [{}]. Load weights [{}]'.format(architecture_file, weights_file))
    else:
        print('Load architecture [{}]'.format(architecture_file))

    return model

if __name__ == '__main__':
    np.random.seed(12227)

    layers = [53, 60, 67]
    n_comp = 3

    cnn_model = load_model('ResNet20', 'ResNet20')

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    X_train, X_test = X_train.astype('float32') / 255, X_test.astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    #Normalize the data
    x_train_mean = np.mean(X_train, axis=0)
    X_train -= x_train_mean
    X_test -= x_train_mean

    print('Layers {} #Components [{}]'.format(layers, n_comp))

    #Learning rate warmups
    lr = 0.01
    schedule = [(100, 1e-3), (150, 1e-4)]
    lr_scheduler = LearningRateScheduler(init_lr=lr, schedule=schedule)
    callbacks = [lr_scheduler]

    #Frozen the layers of the model to avoid changing in the pre-trained weights
    for i in range(0, len(cnn_model.layers)):
        cnn_model.layers[i].trainable = False

    hyper_net = LatentHyperNet(model=cnn_model,
                               layers=layers,
                               n_comp=n_comp,
                               oaa=True,
                               as_network=True)

    X_train_10p, _, y_train_10p, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=42)
    hyper_net.fit(X_train_10p, y_train_10p)

    #Interpret PLS as fully connected layers, enabling  faster inference
    #See https://github.com/arturjordao/PLSGPU for more details
    if hyper_net.as_network == True:
        cnn_model = hyper_net.model
        H = Dense(512, kernel_regularizer=regularizers.l2(0.0005), name='head1')(cnn_model.output)
        inp = cnn_model.input

    else:
        X_train = hyper_net.transform(X_train)
        X_test = hyper_net.transform(X_test)
        inp = Input((X_train.shape[1],))
        H = Dense(512, kernel_regularizer=regularizers.l2(0.0005), name='head1')(inp)

    #Insert Head (classifier). These weights are trainable
    H = Activation('relu', name='head_2')(H)
    H = BatchNormalization(name='head_3')(H)
    H = Dropout(0.5, name='head_4')(H)
    H = Dense(y_train.shape[1], name='head_5')(H)
    H = Activation('softmax', name='head_6')(H)
    model = Model(inp, H)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(X_train, y_train, verbose=2, epochs=10, batch_size=128, callbacks=callbacks)

    y_pred = model.predict(X_test)

    acc_test = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    print('Accuracy LHN[{:.4f}]'.format(acc_test))
