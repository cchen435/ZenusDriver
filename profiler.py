import argparse
import os
import tensorflow as tf

from util import parse_yaml_config, gen_yaml_config, random
from model import load_model
from monitor import Monitor

def init():
    gpus = tf.config.list_physical_devices('GPU') 
    if not gpus:
        raise RuntimeError("Only support GPU profiling now")

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
    
    if gpus: 
        tf.config.experimental.reset_memory_stats('/gpu:0')
    

def finalize(m: Monitor, rounds: int):
    gpus = tf.config.list_physical_devices('GPU')
    peak_mem_use = tf.config.experimental.get_memory_info('/gpu:0')['peak']
    platform, latency, peak_gpu_util, peak_mem_util = m.stop()
    return (platform, latency/rounds, peak_gpu_util, peak_mem_use, peak_mem_util)

def run(config: dict):
    model_info = config['Model']

    model = load_model(model_info)

    print("\n\nWarmup!\n")
    # warm up
    for round in range(10):
        input = random(model_info['batch_size'], model_info['input_shape'])
        output = model(input)
    
    m = Monitor(1000)

    # profiling 
    print("\n\nProfiling!\n")
    for round in range(1000):
        input = random(model_info['batch_size'], model_info['input_shape'])
        output = model(input)
    
    print("\n\nDone!\n")

    return finalize(m, 1000)


parser = argparse.ArgumentParser(
                    prog = 'Zenus Profiler',
                    description = 'A profiling tool to profile machine learning models',
                    epilog = 'Thanks for choosing Zenus')

parser.add_argument("filename", type=str, help="the yml config file for the job")

if __name__ == "__main__":
    args = parser.parse_args()
    config = parse_yaml_config(args.filename)
    m = init()
    utils = run(config);
    print("\nUtils: ", utils)
    result = {'platform': utils[0], 'latency': utils[1], 'peak_gpu_utils': utils[2], 'peak_mem_utils': utils[3]}

    if hasattr(config, 'Profiling'):
        config['Profilng'].append(result)
    else:
        config['Profiling'] = [result]

    print(args.filename.split('.'))
    dirname = os.path.dirname(args.filename)
    basename = os.path.basename(args.filename)
    name = os.path.splitext(basename)[0]
    filename = os.path.join(dirname, name + "_profile.yml")
    print("config: ", config, ', filename: ', filename)
    gen_yaml_config(config, filename)
