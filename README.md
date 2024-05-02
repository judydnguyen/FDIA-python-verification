# FDIA-python-verification

[![Build Status](https://travis-ci.com/stanleybak/nnenum.svg?branch=master)](https://app.travis-ci.com/github/stanleybak/nnenum)

# nnenum - python verification version of FDIA
**nnenum** (pronounced *en-en-en-um*) is a high-performance neural network verification tool. Multiple levels of abstraction are used to quickly verify ReLU networks without sacrificing completeness. Analysis combines three types of zonotopes with star set (triangle) overapproximations, and uses [efficient parallelized ReLU case splitting](http://stanleybak.com/papers/bak2020cav.pdf). The tool is written in Python 3, uses GLPK for LP solving and directly accepts [ONNX](https://github.com/onnx/onnx) network files and `vnnlib` specifications as input. The [ImageStar trick](https://arxiv.org/abs/2004.05511) allows sets to be quickly propagated through all layers supported by the [ONNX runtime](https://github.com/microsoft/onnxruntime), such as convolutional layers with arbitrary parameters.

### Getting Started
Normal training: `train.ipynb`

Adversarial Training: `adversarial_train.ipynb`

Model Testing:  `test.ipynb`

Verification: 

```chmod +x ./one_model_verification.sh && ./one_model_verification.sh```

### Acknowledgement ###
The following citations can be used for nnenum:

```
@inproceedings{bak2021nfm,
  title={nnenum: Verification of ReLU Neural Networks with Optimized Abstraction Refinement},
  author={Bak, Stanley},
  booktitle={NASA Formal Methods Symposium},
  pages={19--36},
  year={2021},
  organization={Springer}
}
```