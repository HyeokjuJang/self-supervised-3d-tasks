import os
from pathlib import Path

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    Dropout,
    UpSampling2D,
    Input,
    concatenate,
)
from tensorflow.keras.utils import plot_model


def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)


def conv_block(
        inputs,
        use_batch_norm=True,
        dropout=0.3,
        filters=16,
        kernel_size=(3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
):
    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = Dropout(dropout)(c)
    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c


def downconv_model(
        input_shape,
        use_batch_norm=True,
        dropout=0.5,
        dropout_change_per_layer=0.0,
        filters=16,
        num_layers=3,
        pooling=None
):
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv_block(
            inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout
        )
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer

    x = conv_block(
        inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout
    )

    if pooling == "max":
        x = MaxPooling2D()(x)
    elif pooling == "avg":
        x = AveragePooling2D()(x)

    model = Model(inputs=[inputs], outputs=[x])
    return model, [down_layers, filters]


def upconv_model(
        input_shape,
        num_classes=3,
        use_batch_norm=True,
        upsample_mode="deconv",  # 'deconv' or 'simple'
        use_dropout_on_upsampling=True,
        dropout=0.5,
        dropout_change_per_layer=0.0,
        filters=128,
        down_layers=(),
        output_activation="softmax",  # 'sigmoid' or 'softmax'
):
    inp = Input(input_shape)
    inputs = [inp]
    x = inp
    if upsample_mode == "deconv":
        upsample = upsample_conv
    else:
        upsample = upsample_simple
    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0
    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
        c_in = Input((int(x) for x in conv.shape[1:]))
        c_in.trainable = False
        inputs.append(c_in)
        x = concatenate([x, c_in])
        x = conv_block(
            inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout
        )
    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
    model = Model(inputs=inputs, outputs=[outputs])
    return model


def custom_unet(
        input_shape,
        num_classes=1,
        use_batch_norm=True,
        upsample_mode="deconv",  # 'deconv' or 'simple'
        use_dropout_on_upsampling=True,
        dropout=0.3,
        dropout_change_per_layer=0.0,
        filters=16,
        num_layers=3,
        output_activation="sigmoid",  # 'sigmoid' or 'softmax'
):
    downconv, data = downconv_model(
        input_shape,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_change_per_layer=dropout_change_per_layer,
        filters=filters,
        num_layers=num_layers,
    )
    upconv = upconv_model(
        downconv.layers[-1].output_shape[1:],
        num_classes=num_classes,
        use_batch_norm=use_batch_norm,
        upsample_mode=upsample_mode,  # 'deconv' or 'simple'
        use_dropout_on_upsampling=use_dropout_on_upsampling,
        dropout=dropout,
        dropout_change_per_layer=dropout_change_per_layer,
        filters=data[1],
        down_layers=data[0],
        output_activation=output_activation,
    )
    outputs = upconv(inputs=[downconv.output, *reversed(data[0])])
    model = Model(inputs=downconv.inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cunet = custom_unet((128, 128, 1))
    cunet.summary()
    plot_model(
        cunet,
        to_file=Path("~/unet_graph.png").expanduser(),
        show_shapes=True,
        expand_nested=True,
    )
