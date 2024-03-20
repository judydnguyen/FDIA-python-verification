'''
eth vnn benchmark 2020
'''


import argparse
import os
import numpy as np

import sys
sys.path.append("../../nnenum/src")

from nnenum.onnx_network import load_onnx_network
from nnenum.network import nn_flatten

feature_range_path = "../../np_data/feature_ranges.npy"
# Load the feature ranges from the .npy file
loaded_feature_ranges = np.load(feature_range_path)

def rescale_to_original(perturbed_image, min_values, max_values):
    # Rescale the perturbed image back to the original range
    # Rescaled image = perturbed_image * (max - min) + min
    rescaled_image = perturbed_image * (max_values - min_values) + min_values
    return rescaled_image


def load_unscaled_images(filename, specific_image=None, epsilon=0.0):
    '''read images from csv file
    
    if epsilon is set, it gets added to the loaded image to get min/max images
    '''

    image_list = []
    labels = []

    test_data = np.load(filename)
    X_test_loaded = test_data['X']
    y_test_loaded = test_data['y']
    
    # Initialize lists to store the min and max for each feature
    min_values = X_test_loaded.min(axis=0)
    max_values = X_test_loaded.max(axis=0)
    ranges = max_values - min_values
    
    total_imgs = X_test_loaded.shape[0]
    print(f"There are total {total_imgs} imgs")
    
    # Normalize data and apply perturbations
    for idx in range(total_imgs):
        if specific_image is not None and idx != specific_image:
            continue
        
        # Exclude constant features from normalization and perturbation
        variable_features = ranges != 0
        features_to_normalize = X_test_loaded[idx, variable_features]
        min_val_var = min_values[variable_features]
        max_val_var = max_values[variable_features]
        
        # Normalize variable features
        # normalized_image = (features_to_normalize - min_val_var) / (max_val_var - min_val_var)
        normalized_image = features_to_normalize
        # Apply epsilon perturbation
        if epsilon != 0.0:
            # perturbation = np.random.uniform(-epsilon, epsilon, normalized_image.shape)
            normalized_image += epsilon
            # Ensure the perturbed data is still in [0, 1] range
            normalized_image = np.clip(normalized_image, -1, 1)
        
        # Rescale variable features back to the original range
        # rescaled_image = rescale_to_original(normalized_image, min_val_var, max_val_var)
        
        # Reconstruct the full image with both variable and constant features
        full_image = np.zeros_like(X_test_loaded[idx])
        full_image[variable_features] = normalized_image
        full_image[~variable_features] = X_test_loaded[idx, ~variable_features]  # Constant features
        
        image_list.append(full_image.reshape(1, -1))
        labels.append(int(y_test_loaded[idx]))
        
    print(f"Processed a total of {len(image_list)} images")
    return np.array(image_list, dtype=np.float32), np.array(labels)

def make_init_box(min_image, max_image):
    'make init box'

    flat_min_image = nn_flatten(min_image)
    flat_max_image = nn_flatten(max_image)

    assert flat_min_image.size == flat_max_image.size

    box = list(zip(flat_min_image, flat_max_image))
        
    return box

def make_init(nn, image_filename, epsilon, specific_image=None):
    'returns list of (image_id, image_data, classification_label, init_star_state, spec)'

    rv = []

    images, labels = load_unscaled_images(image_filename, specific_image=specific_image)
    min_images, _ = load_unscaled_images(image_filename, specific_image=specific_image, epsilon=-epsilon)
    max_images, _ = load_unscaled_images(image_filename, specific_image=specific_image, epsilon=epsilon)
    
    print("making init states")
    # print(len(images))
    correct_cls = False
    with open(f"../examples/fdia/image_{specific_image if specific_image else 0}_epsilon_{epsilon}.vnnlib", 'w+') as f:
        for image_id, (image, classification) in enumerate(zip(images, labels)):
            output = nn.execute(image)
            flat_output = nn_flatten(output)

            num_outputs = flat_output.shape[0]
            label = np.argmax(flat_output)

            if label == labels[image_id]:
                # correctly classified
                correct_cls = True
                min_image = min_images[image_id]
                max_image = max_images[image_id]

                init_box = make_init_box(min_image, max_image)
                f.write(f"; FFNN benchmark image {specific_image} with epsilon = {epsilon}\n\n")

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
                    if i == classification:
                        continue

                    f.write(f"    (and (>= Y_{i} Y_{classification}))\n")

                f.write("))")

                break
    if correct_cls:
        print(f"Writing successfully to {f'../examples/fdia/image_{specific_image if specific_image else 0}_epsilon_{epsilon}.vnnlib'}!")
        f.close()
    else:
        f_path = f.name  # get the path to the file before closing
        f.close()  # make sure to close the file before attempting to delete it
        os.remove(f_path)  # delete the file
        print(f"Removed file {f_path}")
    return rv

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Epsilon for perturbation')
    parser.add_argument('--img_idx', type=int, default=0,
                        help='image for perturbation')
    args = parser.parse_args()
    epsilon = args.epsilon

    onnx_filename = f'../../fdia_model_ffnn_pytorch.onnx'
    image_filename = '../../np_data/test_data_01.npz'

    onnx_filename = f'{onnx_filename}'

    # nn = load_onnx_network(onnx_filename)
    # print(f"loading onnx network from {onnx_filename}")
    nn = load_onnx_network(onnx_filename)
    print(f"loaded network with {nn.num_relu_layers()} ReLU layers and {nn.num_relu_neurons()} ReLU neurons")

    specific_image = args.img_idx if args.img_idx else None
    print("Loading images...")
    tup_list = make_init(nn, image_filename, epsilon, specific_image=specific_image)
    print(f"made {len(tup_list)} init states")


if __name__ == '__main__':
    main()
 
