import marimo

__generated_with = "0.7.17"
app = marimo.App(width="medium")


@app.cell
def __():
    import numpy as np
    import tensorflow as tf
    return np, tf


@app.cell
def __(tf):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="ad01_int8.tflite", experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()

    # Get input and output details.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, interpreter, output_details


@app.cell
def __(input_details, np):
    input_shape = input_details[0]['shape']
    print('input shape: ', input_shape)
    np.random.seed(0)
    random_input = np.random.randint(-127, 128, size=(1, 640), dtype=np.int8)
    print('input: \n', random_input[:5,:5])
    return input_shape, random_input


@app.cell
def __(input_details, interpreter, random_input):
    interpreter.set_tensor(input_details[0]['index'], random_input)
    interpreter.invoke()
    return


@app.cell
def __(interpreter):
    for detail in interpreter.get_tensor_details():
        print(detail)
    return detail,


@app.cell
def __(interpreter, np):
    # Layer 1:
    input_index = 0
    weights_index = 11
    input = interpreter.get_tensor(input_index)
    weight1 = interpreter.get_tensor(weights_index)
    print('input shape: ', interpreter.get_tensor(input_index).shape)
    print('input: \n', interpreter.get_tensor(input_index)[:5,:5])
    print('weights shape: ', interpreter.get_tensor(weights_index).shape)
    print('weight: \n', interpreter.get_tensor(weights_index)[:5,:5])
    out1 = np.matmul(input.astype(np.int32), weight1.astype(np.int32).transpose())
    print('out1: \n', out1[:5, :5])
    return input, input_index, out1, weight1, weights_index


@app.cell
def __(interpreter, out1):
    # Layer 2:
    out2 = interpreter.get_tensor(1) + out1
    print('out2: \n', out2[:5, :5])

    return out2,


@app.cell
def __():
    # Layer 3
    return


if __name__ == "__main__":
    app.run()
