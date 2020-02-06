import functools
import glob
import time
from os.path import expanduser
import _thread

import absl.flags as flags
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from self_supervised_3d_tasks.algorithms import patch3d_utils

from self_supervised_3d_tasks.algorithms import patch_utils
from self_supervised_3d_tasks.data.data_generator import get_data_generators

from self_supervised_3d_tasks.algorithms.patch_model_preprocess import get_crop_patches_fn
from self_supervised_3d_tasks.custom_preprocessing.retina_preprocess import apply_to_x
from self_supervised_3d_tasks.data.kaggle_retina_data import KaggleGenerator
from self_supervised_3d_tasks.data.nifti_loader import DataGeneratorUnlabeled3D
from self_supervised_3d_tasks.datasets import get_data
from self_supervised_3d_tasks.custom_preprocessing.cpc_preprocess import preprocess, preprocess_grid
from self_supervised_3d_tasks.keras_algorithms import cpc
from self_supervised_3d_tasks.keras_algorithms.custom_utils import load_permutations_3d
from self_supervised_3d_tasks.keras_algorithms.jigsaw import get_training_preprocessing
from self_supervised_3d_tasks.preprocess import get_crop, get_random_flip_ud, get_drop_all_channels_but_one_preprocess, \
    get_pad

import seaborn as sns

def plot_sequences(x, y, labels=None, output_path=None):
    ''' Draws a plot where sequences of numbers can be studied conveniently '''

    images = np.concatenate([x, y], axis=1)
    n_batches = images.shape[0]
    n_terms = images.shape[1]
    counter = 1
    for n_b in range(n_batches):
        for n_t in range(n_terms):
            plt.subplot(n_batches, n_terms, counter)
            plt.imshow(images[n_b, n_t, :, :, :])
            plt.axis('off')
            counter += 1
        if labels is not None:
            plt.title(labels[n_b], fontdict={'color': 'white'})

    if output_path is not None:
        plt.savefig(output_path, dpi=600)
    else:
        plt.show()


def get_lena_numpy():
    im_frame = Image.open('data_util/resources/lena.jpg')

    im_frame.load()
    im_frame = im_frame.resize((300,300))

    img = np.asarray(im_frame, dtype="float32")
    img /= 255

    return img

def get_lena():
    img = img = tf.io.read_file('data_util/resources/lena.jpg')
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return {"image": tf.image.resize(img, [300, 300])}

def test_brain():
    params = {
        'dataset': 'ukb3d',
        'preprocessing': [],
        'dataset_dir': "/mnt/mpws2019cl1/brain_mri/tf_records",
    }

    f = functools.partial(
        get_data,
        split_name='train',
        is_training=True,
        num_epochs=1,
        shuffle=False,
        drop_remainder=True,
        **params)

    result = f({'batch_size': 1})
    iterator = result.make_one_shot_iterator()
    i = 0

    with tf.Session() as sess:
        tf.logging.set_verbosity(tf.logging.INFO)

        while True:
            el = iterator.get_next()
            i += 1

            batch = sess.run(el)
            print("{} iteration, shape: {}".format(i, batch["image"].shape))

def count_rec():
    tf.enable_eager_execution()
    files = glob.glob("/mnt/mpws2019cl1/brain_mri/tf_records/*.tfrecord*")

    for f in files:
        print(f+str(sum(1 for _ in tf.data.TFRecordDataset(f))))

def test_records():
    tf.enable_eager_execution()

    print("starting")
    files = glob.glob("/mnt/mpws2019cl1/brain_mri/tf_records/*.tfrecord*")

    print(files)

    filesSize = len(files)
    cnt = 0

    raw_dataset = tf.data.TFRecordDataset(files)

    for raw_record in raw_dataset.take(10):
        print(repr(raw_record))

    # Create a description of the features.

    IMAGE_KEY = "image/encoded"
    HEIGHT_KEY = "image/height"
    WIDTH_KEY = "image/width"
    DEPTH_KEY = "image/depth"
    CHANNELS_KEY = "image/channels"

    FEATURE_MAP = {
        IMAGE_KEY: tf.FixedLenFeature(shape=[128, 128, 128, 2], dtype=tf.float32),
        HEIGHT_KEY: tf.FixedLenFeature(shape=[], dtype=tf.int64),
        WIDTH_KEY: tf.FixedLenFeature(shape=[], dtype=tf.int64),
        DEPTH_KEY: tf.FixedLenFeature(shape=[], dtype=tf.int64),
        CHANNELS_KEY: tf.FixedLenFeature(shape=[], dtype=tf.int64),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, FEATURE_MAP)

    parsed_dataset = raw_dataset.map(_parse_function)
    for i in parsed_dataset:
        print(i)
    print("done")

def test_kaggle_retina():
    params = {
        'dataset': 'kaggle_retina',
        'preprocessing': [],
        'dataset_dir': "/mnt/mpws2019cl1/kaggle_retina/tf_records",
    }

    f = functools.partial(
        get_data,
        split_name='train',
        is_training=True,
        num_epochs=1,
        shuffle=False,
        drop_remainder=True,
        **params)

    result = f({'batch_size': 1})
    iterator = result.make_one_shot_iterator()
    el = iterator.get_next()

    with tf.Session() as sess:
        batch = sess.run(el)
        print(batch["data"].shape)


def test_mnist_data_generator():
    flags.DEFINE_string('dataset', 'cpc_test', 'Which dataset to use, typically '
                                               '`imagenet`.')

    flags.DEFINE_string('dataset_dir', 'data_util/tf_records', 'Location of the dataset files.')
    flags.DEFINE_string('preprocessing', None, "")
    flags.DEFINE_integer('random_seed', 1, "")

    f = functools.partial(
        get_data,
        split_name='train',
        is_training=True,
        num_epochs=1,
        shuffle=False,
        drop_remainder=True)

    result = f({'batch_size': 8})
    print(result)

    iterator = result.make_one_shot_iterator()
    el = iterator.get_next()
    with tf.Session() as sess:
        batch = sess.run(el)
        print(batch["example"]["image/encoded"].shape)

        plot_sequences(batch["example"]["image/encoded"], batch["example"]["image/encoded_pred"],
                       batch["example"]["image/labels"], output_path=r'testXXX.png')


def show_batch(image_batch):
    plt.figure(figsize=(10, 10))
    dim = int(np.sqrt(len(image_batch)))

    if dim * dim < len(image_batch):
        dim += 1

    for n in range(len(image_batch)):
        ax = plt.subplot(dim, dim, n + 1)
        plt.imshow(image_batch[n])
        plt.axis('off')

    plt.show()

def show_batch_numpy(image_batch):
    length = image_batch.shape[0]

    plt.figure(figsize=(10, 10))
    dim = int(np.sqrt(length))

    if dim * dim < length:
        dim += 1

    for n in range(length):
        ax = plt.subplot(dim, dim, n + 1)
        plt.imshow(image_batch[n,:,:,:])
        plt.axis('off')

    plt.show()


def show_img(img):
    print(img.shape)

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()


def chain(f, g):
    return lambda x: g(f(x))


def test_preprocessing_baseline():
    gen = KaggleGenerator(batch_size=36, shuffle=False, categorical=False,
                          pre_proc_func_train = apply_to_x)
    show_batch(gen[0][0])

    # sns.distplot(img[:,:,0].flatten())
    # plt.show()
    # sns.distplot(img[:,:,1].flatten())
    # plt.show()
    # sns.distplot(img[:,:,2].flatten())
    # plt.show()

def test_preprocessing_cpc():
    lena = get_lena_numpy()

    pp = preprocess([lena], 256, 7)

    patches = preprocess_grid(pp)
    print(patches[0][0].shape)

    show_batch(pp[0])

    show_batch(patches[0][1][0])
    show_batch(patches[0][1][1])

    show_batch(patches[0][0][4])

def test_cpc_gen():
    gen = cpc.get_training_generators(1, "kaggle_retina")
    data = gen[0][0][0][1]

    print(data.shape)
    print(data[0].max())
    print(data[0].min())

    show_batch(data[0])

# def test_data_jigsaw():
#     gen = get_training_generators(1, "kaggle_retina")
#     data = gen[0][0][0]
#
#     print(data.shape)
#     print(data[0].max())
#     print(data[0].min())
#
#     show_batch(data[0])
#
# def test_pil_fit():
#     x, _1, _2 = get_finetuning_generators(1, "kaggle_retina", training_proportion=0.8)
#     show_batch(x[0][0][0])


def test_preprocessing():
    with tf.Session() as sess:
        f = get_crop(is_training=True, crop_size=(256, 256))
        # f = chain(f, get_random_flip_ud(is_training=True)) also for new version?
        # f = get_crop_patches_fn(is_training=True, split_per_side=7, patch_jitter=-32)
        f = chain(f, get_crop_patches_fn(is_training=True, split_per_side=7, patch_jitter=-32))

        f = chain(f, get_random_flip_ud(is_training=True))
        f = chain(f, get_crop(is_training=True, crop_size=(56, 56)))
        f = chain(f, get_drop_all_channels_but_one_preprocess())
        f = chain(f, get_pad([[4, 4], [4, 4], [0, 0]], "REFLECT"))

        patches = sess.run(f(get_lena()))
        print(patches["image"].shape)
        show_batch(patches["image"])


def plot_3d(image, dim_to_animate):
    n = len(image)
    ax = []
    frame = []
    ani = []

    for i in range(n):
        ax.append(plt.subplot(n, 1, i+1))
        frame.append(None)
        ani.append(-1)

    while True:
        for i in range(n):
            img = image[i]
            ani[i] += 1

            if ani[i] >= img.shape[dim_to_animate]:
                ani[i] = 0

            idx = [ani[i] if dim == dim_to_animate else slice(None) for dim in range(img.ndim)]
            im = np.squeeze(img[idx], axis=2)

            if frame[i] is None:
                frame[i] = ax[i].imshow(im, cmap="binary")
            else:
                frame[i].set_data(im)

        time.sleep(0.05)
        plt.pause(.1)
        plt.draw()

def test_xxx():
    trainp, valp = get_training_preprocessing()

    x, _ = get_data_generators("/mnt/mpws2019cl1/Task02_Heart", data3d=True,
                               test_data_generator_args={"dim": (128, 128, 128),
                                                         "pre_proc_func": valp},
                               train_data_generator_args={"dim": (128, 128, 128),
                                                          "pre_proc_func": trainp})

    print(x[0][0][0].shape)

    plot_3d(x[0][0][0][0:4], 0)


if __name__ == "__main__":
    x = load_permutations_3d()
    print(x)