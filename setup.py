from setuptools import setup, find_packages

setup(
    name="self-supervised-3d-tasks",
    version="0.0.1",
    packages=find_packages(),

    package_data={
        'permutations': ['*.bin'],
    }, install_requires=['scikit-image', 'joblib', 'numpy==1.19.3', 'nibabel', 'scipy', 'pillow', 'pandas',
                         'matplotlib', 'seaborn', 'albumentations', 'tqdm', 'pydot', 'tensorflow-gpu==2.2.0', 'scikit-learn', 'hyperopt',
                         'tensorflow_addons', 'h5py==2.10.0']
)
