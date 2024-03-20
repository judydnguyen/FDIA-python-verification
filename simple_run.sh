#!/bin/bash -e

cd src

# python3 ../examples/fdia/generate_spec.py --epsilon 0.01 --img_idx 2
# python3 -m nnenum.test ../examples/fdia/fdia_model_ffnn_pytorch.onnx ../examples/fdia/image_2_epsilon_0.01.vnnlib 60  ../examples/fdia/idx_2_epsilon_0.01.txt

# python3 ../examples/fdia/generate_spec.py --epsilon 0.01 --img_idx 11
# python3 ../examples/fdia/generate_spec.py --epsilon 0.01 --img_idx 12
# python3 ../examples/fdia/generate_spec.py --epsilon 0.01 --img_idx 13
# python3 -m nnenum.test ../examples/fdia/fdia_model_ffnn_pytorch.onnx ../examples/fdia/image_10_epsilon_0.01.vnnlib 60  ../examples/fdia/idx_10_epsilon_0.01.txt

# python3 -m nnenum.test ../examples/fdia/fdia_model_ffnn_pytorch.onnx ../examples/fdia/image_11_epsilon_0.01.vnnlib 60  ../examples/fdia/idx_11_epsilon_0.01.txt
# python3 -m nnenum.test ../examples/fdia/fdia_model_ffnn_pytorch.onnx ../examples/fdia/image_12_epsilon_0.01.vnnlib 60  ../examples/fdia/idx_12_epsilon_0.01.txt
# python3 -m nnenum.test ../examples/fdia/fdia_model_ffnn_pytorch.onnx ../examples/fdia/image_13_epsilon_0.01.vnnlib 60  ../examples/fdia/idx_13_epsilon_0.01.txt

python3 ../examples/fdia/generate_massive_spec.py -e 0.01 -n 10
cd ..