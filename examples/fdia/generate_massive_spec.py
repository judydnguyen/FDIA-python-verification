import argparse
from collections import defaultdict
import os
from random import shuffle
import numpy as np

import sys
sys.path.append("../src")

import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from nnenum.onnx_network import load_onnx_network
from nnenum.network import nn_flatten

def rescale_to_original(perturbed_image, min_values, max_values):
    # Rescale the perturbed image back to the original range
    # Rescaled image = perturbed_image * (max - min) + min
    rescaled_image = perturbed_image * (max_values - min_values) + min_values
    return rescaled_image

def load_unscaled_images(img, epsilon=0.0, variable_features=[]):
    '''read images from csv file
    
    if epsilon is set, it gets added to the loaded image to get min/max images
    '''
    features_to_normalize = img[variable_features]
    normalized_image = features_to_normalize
    full_image = img
    # Apply epsilon perturbation
    if abs(epsilon) > 0:
        # perturbation = np.random.uniform(-epsilon, epsilon, normalized_image.shape)
        normalized_image += epsilon
        # print(f"Ensure the perturbed data is still in [-1, 1] range")
        normalized_image = np.clip(normalized_image, -1, 1)
        
        # Rescale variable features back to the original range
        # rescaled_image = rescale_to_original(normalized_image, min_val_var, max_val_var)
        
        # Reconstruct the full image with both variable and constant features
        full_image = np.zeros_like(img)
        full_image[variable_features] = normalized_image
        full_image[~variable_features] = img[~variable_features]  # Constant features
        
    # image_list.append(full_image.reshape(1, -1))
    # labels.append(int(y_test_loaded[idx]))
        
    # print(f"Processed a total of {len(image_list)} images")
    return np.array(full_image, dtype=np.float32)

def make_init_box(min_image, max_image):
    'make init box'

    flat_min_image = nn_flatten(min_image)
    flat_max_image = nn_flatten(max_image)

    assert flat_min_image.size == flat_max_image.size

    box = list(zip(flat_min_image, flat_max_image))
        
    return box

def make_init(nn, img, epsilon, 
              img_id = 0, gt_label = 0, 
              variable_features=[]):
    'returns list of (image_id, image_data, classification_label, init_star_state, spec)'

    rv = []

    image = load_unscaled_images(img, epsilon = 0, variable_features=variable_features)
    min_image = load_unscaled_images(img, epsilon=-epsilon, variable_features=variable_features)
    max_image = load_unscaled_images(img, epsilon=epsilon, variable_features=variable_features)
    
    # import IPython
    # IPython.embed()
    print("making init states")
    # print(len(images))
    correct_cls = False
    os.makedirs(f"../examples/fdia/epsilon_{epsilon}/class_{int(gt_label)}", exist_ok=True)
    with open(f"../examples/fdia/epsilon_{epsilon}/class_{int(gt_label)}/image_{img_id}_epsilon_{epsilon}.vnnlib", 'w+') as f:
        output = nn.execute(image)
        flat_output = nn_flatten(output)

        num_outputs = flat_output.shape[0]
        label = np.argmax(flat_output)

        if label == gt_label:
            # correctly classified
            correct_cls = True
            min_image = min_image
            max_image = max_image

            init_box = make_init_box(min_image, max_image)
            f.write(f"; FFNN benchmark image {img_id} with epsilon = {epsilon}\n\n")

            for i in range(len(init_box)):
                f.write(f"(declare-const X_{i} Real)\n")

            f.write("\n")
                
            for i in range(2):
                f.write(f"(declare-const Y_{i} Real)\n")

            f.write("\n; Input constraints:\n")

            for i, (lb, ub) in enumerate(init_box):
                f.write(f"(assert (<= X_{i} {ub:.18f}))\n")
                f.write(f"(assert (>= X_{i} {lb:.18f}))\n\n")

            f.write("\n; Output constraints:\n")
            f.write("(assert (or\n")

            for i in range(num_outputs):
                if i == gt_label:
                    continue

                f.write(f"    (and (>= Y_{i} Y_{gt_label}))\n")

            f.write("))")

    if correct_cls:
        print(f"Writing successfully to {f'../examples/fdia/epsilon_{epsilon}/class_{int(gt_label)}/image_{img_id}_epsilon_{epsilon}.vnnlib'}!")
        f.close()
        rv.append(image)
    else:
        f_path = f.name  # get the path to the file before closing
        f.close()  # make sure to close the file before attempting to delete it
        os.remove(f_path)  # delete the file
        print(f"Removed file {f_path}")
    return rv

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e', '--epsilon', type=float, default=0.1,
                    help='Epsilon for perturbation')
    parser.add_argument('-n', '--num_per_class', type=int, default=10,
                    help='Number of samples for each class')
    parser.add_argument('-m', '--model_path', type=str, default="",
                    help='Path to onnx model')
    parser.add_argument('-s', '--seed', type=int, default=10,
                    help='Seeding constant')
    parser.add_argument('-f', '--filename', type=str, default='../np_data/test_data_01.npz',
                help='Number of samples for each class')
    args = parser.parse_args()
    
    epsilon = args.epsilon
    num_per_class = args.num_per_class
    logger.info(f"Loading {num_per_class} per each class for verification, epsilon {epsilon}...")
    
    image_list = []
    labels = []

    test_data = np.load(args.filename)
    X_test_loaded = test_data['X']
    labels = test_data['y']
    
    min_values = X_test_loaded.min(axis=0)
    max_values = X_test_loaded.max(axis=0)
    ranges = max_values - min_values
    
    discrete_columns = [28, 57, 86, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
    
    variable_features = ranges != 0

    # Create a boolean mask to remove the discrete columns from variable_features
    variable_features_mask = ~np.isin(np.arange(len(variable_features)), discrete_columns)

    # Apply the mask to get the updated variable_features
    variable_features = variable_features & variable_features_mask

    print(f"Poisoning the features: {variable_features}")
    
    total_imgs = X_test_loaded.shape[0]
    print(f"There are total {total_imgs} imgs")
    
    unique_classes = np.unique(labels)
    class_indices = defaultdict(list)

    # Collect indices for each class
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
        
    selected_indices = []
    for label in unique_classes:
        # Shuffle indices for the current class to ensure randomness
        np.random.shuffle(class_indices[label])
        # Choose 'num_per_class' indices
        # selected_indices.extend(shuffled_indices[:num_per_class])

    onnx_filename = f'../examples/fdia/fdia_model_ffnn_pytorch.onnx' if not args.model_path else args.model_path
    image_filename = '../np_data/test_data_01.npz'

    onnx_filename = f'{onnx_filename}'
    nn = load_onnx_network(onnx_filename)
    print(f"loaded network with {nn.num_relu_layers()} ReLU layers and {nn.num_relu_neurons()} ReLU neurons")
    print(f"unique_classes: {unique_classes}")
    # return
    for label in unique_classes:
        img_idx = 0
        sample_idx = img_idx
        while(img_idx < num_per_class):
            specific_image = class_indices[label][sample_idx]
            img = X_test_loaded[specific_image]
            img_label = labels[specific_image]
            tup_list = make_init(nn, img, epsilon, img_idx, 
                                 img_label, variable_features)
            sample_idx += 1
            if len(tup_list):
                img_idx += 1
    print(f"made {int((label+1)*num_per_class)} init states")
        
if __name__ == "__main__":
    main()