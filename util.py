import yaml
import tensorflow as tf

def random(batch_size, input_shape: list):
    shape = [batch_size] + input_shape
    return tf.random.normal(shape, mean=0.0)

def gen_yaml_config(data : dict, config_file : str):
    with open(config_file, "w") as f:
        yaml.dump(data, f)

def parse_yaml_config(config_file : str):
    with open("config.yml", "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print("Failed to load yaml file {} with error: {}".format(config_file, e))

if __name__ == "__main__":
    config = {'Model' : {'type': 'keras', 'path': 'resnet.ResNet50', 'batch_size' : 4, 'mode': 'inference', 'input_shape': [224, 224, 3]}, 
              'QoS':{'latency': '4'}, 
              'Profiling': [{'Platform': 'ATS', 'latency': 34, 'peak_mem_utilization': 75, 'peak_gpu_utilization': 50}]}
    gen_yaml_config(config, 'config.yml')
    conf = parse_yaml_config('config.yml')
    print(conf)

    random(conf['Model']['batch_size'], conf['Model']['input_shape'])