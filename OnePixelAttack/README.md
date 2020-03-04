## Reqirements
* pytorch
* numpy
* scipy.fftpack

## Train

The multi-channel architecture with a randomized diversification can be trained for any type of deep classifiers and suits for any training data.
  
    $ python train_model_multi_channel.py --model="vgg16" --permut=9 --epochs=1000 --lr=1e-3 --batch_size=128  

## Test

For the test the adversarial examples were generate by using the attack propodsed by 
> Jiawei Su, Danilo Vasconcellos Vargas, Sakurai Kouichi:  
> [One pixel attack for fooling deep neural networks](https://arxiv.org/pdf/1710.08864.pdf) 

The python attacks implementation was taken from https://github.com/DebangLi/one-pixel-attack-pytorch
