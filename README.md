# A PyTorch implement of MemN2N

This is a simple re-implement of [MemN2N](https://github.com/facebook/MemNN) using PyTorch. We mainly refer to its TensorFlow [implement](https://github.com/domluna/memn2n). It is only used for academic research or study. If you have any question about the model, please refer to the [paper](https://arxiv.org/abs/1503.08895). If you have any question about this implement, please feel free to contact me.

## Getting Started

This project is implemented using Python 2.7, PyTorch (Version 0.3.0.post4) and Eclipse (Version: Oxygen Release (4.7.0)). The dataset comes from [bAbl](http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz). A sample data of en Task 1 is integrated with this project for convenience.

The main steps to run this project include:

* Download the code and data.
* Import them into eclipse.
* Run single.py

## Notes

The features that have been implemented:

* Bag-of-words (Bow)
* Position encoding (PE)
* Hops
* Adjacent weight tying
* Non-linearity
* Random injection noise (RN)

The features that haven't been implemented:

* Linear start training (LS)
* RNN-style layer-wise weight tying (LW)
* Joint training

