import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="ad01_int8.tflite", experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()

# Get input and output details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Generate a random 8x640 int8 input.
input_shape = input_details[0]['shape']
random_input = np.random.randint(-128, 127, size=(1, 640), dtype=np.int8)

# Set the tensor for the input.
interpreter.set_tensor(input_details[0]['index'], random_input)

# Run the model.
interpreter.invoke()

# Function to retrieve layer outputs by their tensor index
def get_intermediate_outputs(interpreter):
    layer_outputs = {}
    for i, tensor_detail in enumerate(interpreter.get_tensor_details()):
        tensor_name = tensor_detail['name']
        tensor_index = tensor_detail['index']
        tensor_shape = tensor_detail['shape']
        tensor_output = interpreter.get_tensor(tensor_index)
        layer_outputs[tensor_name] = tensor_output
        print(f"Layer {i}: {tensor_name} | Shape: {tensor_shape} | Output: {tensor_output}\n")
    return layer_outputs

# Get the output from every layer.
intermediate_outputs = get_intermediate_outputs(interpreter)

breakpoint()
print('finished')
