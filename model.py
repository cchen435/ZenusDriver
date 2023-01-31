import tensorflow as tf
from tensorflow import keras

def load_keras_model(path):
    items = path.split(".")
    obj = tf.keras.applications
    for component in items:
        if hasattr(obj, component):
            obj = getattr(obj, component)
        else:
            raise ValueError("No attribute {} found in obj {}".format(component, obj))
    return obj()
        

def load_model(model_info: dict):
    if model_info['type'] == 'keras':
        return load_keras_model(model_info['path']);
    else:
        raise("Currently only supports loading models from keras. Other sources are not supported yet !")


