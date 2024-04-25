#!/bin/bash -e
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
cd src

MODEL_PATH="../onnx_models/fdia_model_ffnn_pytorch_50_100_50.onnx"
python3 ../examples/fdia/generate_massive_spec.py -e 0.01 -n 20 -m $MODEL_PATH
python3 ../examples/fdia/generate_massive_spec.py -e 0.025 -n 20 -m $MODEL_PATH
python3 ../examples/fdia/generate_massive_spec.py -e 0.005 -n 20 -m $MODEL_PATH

python3 -m nnenum.test $MODEL_PATH ../examples/fdia/epsilon_0.01/class_0 60 ../examples/fdia/results/class_0 1 auto &&
python3 -m nnenum.test $MODEL_PATH ../examples/fdia/epsilon_0.01/class_1 60 ../examples/fdia/results/class_1 1 auto &&
python3 -m nnenum.test $MODEL_PATH ../examples/fdia/epsilon_0.005/class_0 60 ../examples/fdia/results/class_0 1 auto &&
python3 -m nnenum.test $MODEL_PATH ../examples/fdia/epsilon_0.005/class_1 60 ../examples/fdia/results/class_1 1 auto &&
python3 -m nnenum.test $MODEL_PATH ../examples/fdia/epsilon_0.025/class_0 60 ../examples/fdia/results/class_0 1 auto &&
python3 -m nnenum.test $MODEL_PATH ../examples/fdia/epsilon_0.025/class_1 60 ../examples/fdia/results/class_1 1 auto
