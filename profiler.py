import argparse
import sys
import tensorflow as tf

from util import parse_yaml_config, random
from model import load_model

def init():
    gpus = tf.config.list_physical_devices('GPU') 
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.reset_memory_stats(gpu)

    except RuntimeError as e:
        print(e)
    

def finalize():
    device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu:0'
    peak_mem_utils = tf.config.experimental.get_memory_info(device)['peak']
    device_details = tf.config.experimental.get_device_details(device)
    print(device_details)

def run(config_file: str):
    config = parse_yaml_config(config_file)
    model_info = config['Model']

    init()

    model = load_model(model_info)
    input = random(model_info['batch_size'], model_info['input_shape'])
    output = model(input)

    finalize()


parser = argparse.ArgumentParser(
                    prog = 'Zenus Profiler',
                    description = 'A profiling tool to profile machine learning models',
                    epilog = 'Thanks for choosing Zenus')

parser.add_argument("filename", type=str, help="the yml config file for the job")

if __name__ == "__main__":
    args = parser.parse_args()
    run(args.filename);
