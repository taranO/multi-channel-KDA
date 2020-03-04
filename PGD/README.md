## Reqirements
* pytorch
* foolbox
* numpy
* scipy.fftpack

## Train

The PGD transferability attack in gray-box scenario is implemented for the VGG16 and ResNet18 single-layer (vanilla classier) and proposed multi-channel (with KDA) models: 
  
    $ python attack_single_model.py --model="vgg16" --epochs=1000 --epsilon=0.5 --iterations=100 --samples=1000

The training of the vanilla and multi-channel (with KDA) mosels are given in OnePixelAttack https://github.com/taranO/multi-channel-KDA/tree/master/OnePixelAttack.

## Test

For the test the adversarial examples were generate by using the attack propodsed by 
> Madry, A., Makelov, A., Schmidt, L., Tsipras, D., Vladu, A.:  
> [Towards deep learning models resistant to adversarial attacks](https://arxiv.org/pdf/1706.06083.pdf) 

The python attacks implementation was taken from https://foolbox.readthedocs.io/en/stable/
