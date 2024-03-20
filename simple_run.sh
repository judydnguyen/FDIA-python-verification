#!/bin/bash -e
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
cd src

# Create the directory if it does not exist
mkdir -p ../examples/fdia/results/class_0

# Run the Python script
# python3 ../examples/fdia/generate_massive_spec.py -e 0.01 -n 10
python3 -m nnenum.test ../examples/fdia/fdia_model_ffnn_pytorch.onnx ../examples/fdia/class_0/image_0_epsilon_0.01.vnnlib 60 ../examples/fdia/results/class_0/image_0_epsilon_0.01.vnnlib.txt
cd ..