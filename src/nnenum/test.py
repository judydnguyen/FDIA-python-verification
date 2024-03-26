'''
nnenum vnnlib front end

usage: "python3 nnenum.py <onnx_file> <vnnlib_file> [timeout=None] [outfile=None]"

Stanley Bak
June 2021
'''

import os
import sys
sys.path.append(".")

import numpy as np
from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.result import Result
from nnenum.onnx_network import load_onnx_network_optimized, load_onnx_network
from nnenum.specification import Specification, DisjunctiveSpec
from nnenum.vnnlib import get_num_inputs_outputs, read_vnnlib_simple

def make_spec(vnnlib_filename, onnx_filename):
    '''make Specification

    returns a pair: (list of [box, Specification], inp_dtype)
    '''

    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(onnx_filename)
    vnnlib_spec = read_vnnlib_simple(vnnlib_filename, num_inputs, num_outputs)

    rv = []

    for box, spec_list in vnnlib_spec:
        if len(spec_list) == 1:
            mat, rhs = spec_list[0]
            spec = Specification(mat, rhs)
        else:
            spec_obj_list = [Specification(mat, rhs) for mat, rhs in spec_list]
            spec = DisjunctiveSpec(spec_obj_list)

        rv.append((box, spec))

    return rv, inp_dtype

def set_control_settings():
    'set settings for smaller control benchmarks'

    Settings.TIMING_STATS = False
    Settings.PARALLEL_ROOT_LP = False
    Settings.SPLIT_IF_IDLE = False
    Settings.PRINT_OVERAPPROX_OUTPUT = False
    Settings.TRY_QUICK_OVERAPPROX = True

    Settings.CONTRACT_ZONOTOPE_LP = True
    Settings.CONTRACT_LP_OPTIMIZED = True
    Settings.CONTRACT_LP_TRACK_WITNESSES = True

    Settings.OVERAPPROX_BOTH_BOUNDS = False

    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
    Settings.OVERAPPROX_GEN_LIMIT_MULTIPLIER = 1.5
    Settings.OVERAPPROX_LP_TIMEOUT = 0.02
    Settings.OVERAPPROX_MIN_GEN_LIMIT = 70

def set_exact_settings():
    'set settings for smaller control benchmarks'

    Settings.TIMING_STATS = True
    Settings.TRY_QUICK_OVERAPPROX = False

    Settings.CONTRACT_ZONOTOPE_LP = True
    Settings.CONTRACT_LP_OPTIMIZED = True
    Settings.CONTRACT_LP_TRACK_WITNESSES = True

    Settings.OVERAPPROX_BOTH_BOUNDS = False

    Settings.BRANCH_MODE = Settings.BRANCH_EXACT

def set_image_settings():
    'set settings for larger image benchmarks'

    Settings.COMPRESS_INIT_BOX = True
    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
    Settings.TRY_QUICK_OVERAPPROX = False
    
    Settings.OVERAPPROX_MIN_GEN_LIMIT = np.inf
    Settings.SPLIT_IF_IDLE = False
    Settings.OVERAPPROX_LP_TIMEOUT = np.inf
    Settings.TIMING_STATS = True

    # contraction doesn't help in high dimensions
    #Settings.OVERAPPROX_CONTRACT_ZONO_LP = False
    Settings.CONTRACT_ZONOTOPE = False
    Settings.CONTRACT_ZONOTOPE_LP = False

def verify_one_spec(onnx_filename, vnnlib_filename, settings_str, timeout):
    spec_list, input_dtype = make_spec(vnnlib_filename, onnx_filename)
    try:
        network = load_onnx_network_optimized(onnx_filename)
        print(f"load network...: {network}")
    except:
        # cannot do optimized load due to unsupported layers
        network = load_onnx_network(onnx_filename)
        print("load network...")

    result_str = 'none' # gets overridden

    num_inputs = len(spec_list[0][0])
    print(f"num_inputs: {num_inputs}")
    print(f"spec_list.len: {len(spec_list)}")

    if settings_str == "auto":
        if num_inputs < 700:
            set_control_settings()
        else:
            set_image_settings()
    elif settings_str == "control":
        set_control_settings()
    elif settings_str == "image":
        set_image_settings()
    else:
        assert settings_str == "exact"
        set_exact_settings()
    print(f"Set settings_str: {settings_str}")
    for init_box, spec in spec_list:
        init_box = np.array(init_box, dtype=input_dtype)

        if timeout is not None:
            if timeout <= 0:
                result_str = 'timeout'
                break

            Settings.TIMEOUT = timeout

        res = enumerate_network(init_box, network, spec)
        result_str = res.result_str

        if timeout is not None:
            # reduce timeout by the runtime
            timeout -= res.total_secs

        if result_str != "safe":
            break
    return res, result_str

def main():
    'main entry point'

    # if len(sys.argv) < 3:
    #     print('usage: "python3 nnenum.py <onnx_file> <vnnlib_file> [timeout=None] [outfile=None] [processes=<auto>]"')
    #     sys.exit(1)
    if len(sys.argv) < 3:
        print('usage: "python3 nnenum.py <onnx_file> <vnnlib_path> [timeout=None] [out_path=None] [processes=<auto>]"')
        sys.exit(1)

    onnx_filename = sys.argv[1]
    vnnlib_path = sys.argv[2]
    timeout = None
    out_path = None
    print(f"vnnlib_path: {vnnlib_path}")
    if len(sys.argv) >= 4:
        timeout = float(sys.argv[3])

    if len(sys.argv) >= 5:
        out_path = sys.argv[4]

    if len(sys.argv) >= 6:
        processes = int(sys.argv[5])
        Settings.NUM_PROCESSES = processes

    if len(sys.argv) >= 7:
        settings_str = sys.argv[6]
    else:
        settings_str = "auto"
        
    vnnlib_files = []
    short_files = []
    # Initialize lists to store results and timing information
    results = []
    times = []
    robust_samples = 0
    not_robust_samples = 0
    unknown_samples = 0
    out_path = f"{out_path}/{settings_str}"
    # walking through the vnnlib_path and run the verification
    for dirpath, dirnames, filenames in os.walk(vnnlib_path):
        # Filter and collect full paths of .vnnlib files
        for file in filenames:
            if file.endswith('.vnnlib'):
                full_path = os.path.join(dirpath, file)
                vnnlib_files.append(full_path)
                short_files.append(file)
    # verify each spec
    for idx, vnnlib_filename in enumerate(vnnlib_files):
        res, result_str = verify_one_spec(onnx_filename, vnnlib_filename, settings_str, timeout)
        if result_str == "safe":
            result_str = "holds"
            result_str += f"\n{res.total_secs}"
            robust_samples += 1
            results.append(1)  # Code for robust
        elif "unsafe" in result_str:
            result_str = "violated"
            result_str += f"\nresult.cinput: {res.cinput}\nresult.coutput: {res.coutput}"
            result_str += f"\n{res.total_secs}"
            not_robust_samples += 1 
            results.append(0)  # Code for not robust
        else:
            unknown_samples += 1
            results.append(2)  # Code for unknown
            
        times.append(res.total_secs)
        os.makedirs(out_path, exist_ok=True)
        file_to_save = f"{out_path}/{short_files[idx]}.txt"
        if out_path is not None:
            with open(file_to_save, 'w+') as f:
                f.write(result_str)
        #print(result_str)
        if result_str == 'error':
            sys.exit(Result.results.index('error'))
            
    # Convert results and times to numpy arrays for summary calculations
    results_array = np.array(results)
    times_array = np.array(times)
    N = len(results_array)  # Total number of samples

    # Calculate summary statistics
    totalTime = times_array.sum()
    avgTime = totalTime / N if N > 0 else 0

    # Print results to the screen
    print("======= ROBUSTNESS RESULTS ==========")
    print(f"Number of robust samples = {robust_samples}, equivalent to {100 * robust_samples / N:.2f}% of the samples.")
    print(f"Number of not robust samples = {not_robust_samples}, equivalent to {100 * not_robust_samples / N:.2f}% of the samples.")
    print(f"Number of unknown samples = {unknown_samples}, equivalent to {100 * unknown_samples / N:.2f}% of the samples.")
    print(f"It took a total of {totalTime:.2f} seconds to compute the verification results, an average of {avgTime:.2f} seconds per image")

    # Save results as .npy files or a .npz archive
    # np.save(f'{out_path}/results_verify_fdia_ffnn.npy', results_array)
    # np.save(f'{out_path}/times_verify_fdia_ffnn.npy', times_array)
    np.savez(f'{out_path}/summary_verify_fdia_ffnn.npz', results=results_array, times=times_array)

if __name__ == '__main__':
    main()
