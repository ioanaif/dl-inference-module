# Deep Learning Inference for Fast Simulation Applications 

C++ Inference module for Generative TensorFlow Models 


## How To Run Inference

Download the Tensorflow C API (https://www.tensorflow.org/install/lang_c) and extract its `./lib/` contents to `./modules/all/`


Moreover, you can run inference for your choice of model and energy input:

```sh
cd module/ensemble/
mkdir build
cd build
cmake ..
make .
./dlinf modelChoice energyValue 
```

where `modelChoice` can be either `dcgan` , `cvae`, `ar`

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

In a .txt file save the network details formated as follows:

name of .pb file 
name of .ckpt file
name of input node 
name of label node
name of output node 
input shape
labels shape

An example of such file ```model_latest_params.txt```: 
 
```Python
modelAR_25-40_tf13.pb
PbWO4_sampled_25__40_24.ckpt
model/input_cells
model/input_labels
model_generation_output
20 24 24 24
20 15
```
In case of multiple input nodes (CVAE) append at the end of .txt file a line containing the extra input node name and another one with its shape. 

## Model Integration Helper Functions Details

Helper functions coresponding to each network particularities are required. These should include the following: 

- label transform function returning a vector of size ```input_label_shape``` transforming the GeV energy value to network label 
- depositions transform function describing the transformation function for network output -> energy values 
- cell positions function returning a vector of cell positions 




