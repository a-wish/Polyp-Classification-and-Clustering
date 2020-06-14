from functools import partial

from neupy import layers


__all__ = ('network',)


def network():

    HalfPadConvolution = partial(layers.Convolution, padding='half')

    return layers.join(
        layers.Input((3, 224, 224)),

        HalfPadConvolution((20, 5, 5), name='conv1_1') > layers.Relu(),
        
	HalfPadConvolution((20, 5, 5), name='conv1_2') > layers.Relu(),
        layers.MaxPooling((2, 2)),


        HalfPadConvolution((60, 5, 5), name='conv2_1') > layers.Relu(),

        HalfPadConvolution((60, 5, 5), name='conv2_2') > layers.Relu(),

        layers.MaxPooling((2, 2)),


        HalfPadConvolution((120, 5, 5), name='conv3_1') > layers.Relu(),

        HalfPadConvolution((120, 5, 5), name='conv3_2') > layers.Relu(),

        HalfPadConvolution((150, 5, 5), name='conv3_3') > layers.Relu(),

        HalfPadConvolution((150, 5, 5), name='conv3_4') > layers.Relu(),

        layers.MaxPooling((2, 2)),


        HalfPadConvolution((128, 5, 5), name='conv4_1') > layers.Relu(),

        HalfPadConvolution((128, 5, 5), name='conv4_2') > layers.Relu(),

        HalfPadConvolution((512, 3, 3), name='conv4_3') > layers.Relu(),

        HalfPadConvolution((512, 3, 3), name='conv4_4') > layers.Relu(),

        layers.MaxPooling((2, 2)),




        layers.Reshape(),

        layers.Linear(2000, name='dense_1') > layers.Relu(),

        layers.Dropout(0.5),

        layers.Linear(1000, name='dense_2') > layers.Relu(),

        layers.Dropout(0.5),

        layers.Linear(1000, name='dense_3') > layers.Softmax(),
    )
