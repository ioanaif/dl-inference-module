"""
Convert model.ckpt to model.pb
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import graph_util

# create a session
sess = tf.Session()

# import best model
saver = tf.compat.v1.train.import_meta_graph('checkpoint/PbWO4_sampled_25-40_24_tf13.ckpt.meta') # graph
saver.restore(sess, 'checkpoint/PbWO4_sampled_25-40_24_tf13.ckpt') # variables

#saver = tf.compat.v1.train.import_meta_graph('checkpoint/params_PbWO4.ckpt.meta') # graph
#saver.restore(sess, 'checkpoint/params_PbWO4.ckpt') # variables




# get graph definition
gd = sess.graph.as_graph_def()

# fix batch norm nodes
for node in gd.node:
  node.device = ""
  print(node)
  if node.op == 'RefSwitch':
    node.op = 'Switch'
    for index in range(len(node.input)):
      if 'moving_' in node.input[index]:
        node.input[index] = node.input[index] + '/read'
  elif node.op == 'AssignSub':
    node.op = 'Sub'
    if 'use_locking' in node.attr: del node.attr['use_locking']


# generate protobuf
#converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, ["logits_set"])
tf.train.write_graph(gd, './checkpoint', 'modelAR_25-40_tf13.pb', as_text=False)
