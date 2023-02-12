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
    
    tf.config.experimental.reset_memory_stats('/gpu:0')

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
    

def finalize(m: Monitor, rounds: int):
    gpus = tf.config.list_physical_devices('GPU')
    platform, latency, peak_gpu_util, peak_mem_use = m.stop()
    return (platform, latency/rounds, peak_gpu_util, peak_mem_use)

def run(config: dict):
    model_info = config['Model']

    model = load_model(model_info)

    mode = getattr(model_info, 'mode', 'inference')
    if mode not in ['inference', 'training']:
        raise(f"Unsupported running mode: {mode}")

    # warm up
    print("\n\nWarmup!\n")
    input = random(model_info['batch_size'], model_info['input_shape'])
    if mode == 'inference':
        for round in range(10):
            output = model(input)
    else:
        model.fit(input, input, epochs=10, batch_size=model_info['batch_size'])
    
    m = Monitor(1000)

    # profiling 
    print("\n\nProfiling!\n")
    if model_info['mode'] == 'inference':
        for round in range(1000):
            output = model(input)
    else:
        model.fit(input, input, epochs=1000, batch_size=model_info['batch_size'])
    
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
    init()
    utils = run(config);
    result = {'platform': utils[0], 'latency': utils[1], 'peak_gpu_utils': utils[2], 'peak_mem_utils': utils[3]}

    if hasattr(config, 'Profiling'):
        config['Profilng'].append(result)
    else:
        config['Profiling'] = [result]

    dirname = os.path.dirname(args.filename)
    basename = os.path.basename(args.filename)
    name = os.path.splitext(basename)[0]
    filename = os.path.join(dirname, name + "_profile.yml")
    gen_yaml_config(config, filename)
