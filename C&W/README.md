## Reqirements
* keras
* numpy
* scipy.fftpack

## Train

The multi-channel architecture with KDA can be trained for any type of deep classifiers and suits for any training data.
  
    $ python train_model_multi_channel.py --type=mnist --epochs=1000 --lr=1e-3 --oprimazer="adam" --batch_size=64

## Test

For the test the adversarial examples were generate by using the attack propodsed by 
> Carlini Nicholas and Wagner David:  
> [Towards evaluating the robustness of neural networks](https://arxiv.org/pdf/1608.04644.pdf) 

The python attacks implementation was taken from https://github.com/carlini/nn_robust_attacks
