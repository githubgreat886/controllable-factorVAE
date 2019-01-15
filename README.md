# Environment

Please make sure that following packages are installed

* Python 3.6
* torch 0.4
* matplotlib
* numpy
* tensorboard
* skimage

# Run

To test the experiment code, please run the following command

`python main.py`

To test with chair dataset, please run

`python main.py --dataset chair`

To test other models, please run

`python main.py --model cfactorvae`

More information

`python main.py help`

~~~
usage: main.py [-h] [--dataset DATASET] [--lr L] [--beta B] [--gamma B]
               [--nb-latents N] [--batch-size N] [--epochs N]
               [--device DEVICE] [--path PATH] [--model MODEL]
main.py: error: unrecognized arguments: help
~~~

Dataset can be downloaded at https://www.di.ens.fr/willow/research/seeing3Dchairs/

Please create a dataset folder called './data' can put the dataset there. For example, create a data folder as the following
~~~
.
└── data
    ├── render_chairs
        └── rendered_chairs1
            ├── xxx.png
            ├── ...
       └── rendered_chairs2
            ├── xxx.png
            ├── ...
    └── ...

    ├── render_chairs_test
        └── rendered_chairs1
            ├── xxx.png
            ├── ...
       └── rendered_chairs2
            ├── xxx.png
            ├── ...
    └── ...
~~~


# Trained Model
Each 'model_xxx.pt' file stores the parameters of a trained model over MNIST dataset