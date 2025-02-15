# Deep Learning Inference for Fast Simulation Applications 

C++ Inference Module for Generative TensorFlow Models 

## How To Run Inference

1. Follow the [Inference Library](./inference/README.md) instalation steps. 

2. Run from an existing model (AR, GAN, VAE): 

```sh
cd modules/
mkdir build
cd build
cmake3 .. -DCMAKE_PREFIX_PATH=<PATH_TO_INFERENCE_LIBRARY_INSTALL_PATH>
make 
./EventGeneration
```

## How To Integrate Your Model 

1. Save your input/output node names. For example, given a Python model:

```Python
# Event Data Innputs

x_sample = tf.placeholder(tf.float32, shape=xs, name='input_cells')
y_sample = tf.placeholder(tf.float32, shape=xs[0], name="input_labels")

# Generated Result

generation = tf.add(a, b, name='output_result')
```

2. Store the graph definition in a `.pb` file as well as the latest checkpoint  in `.ckpt` files (`.data`, `.index`, `.meta`)

3. Note your input data shape information (both for samples and labels). 

## Model Integration Info File Example

```Python
modelType       =  "dcgan"
modelGraph      = "../dcgan.pb"
modelRestore    = "../model.b32.ckpt"
inputNode       = "input_cells"
labelNode       = "input_labels"
outputNode      = "output_result"
inputShape      = {64,8,8,24}
labelShape      = {64,100}
```



