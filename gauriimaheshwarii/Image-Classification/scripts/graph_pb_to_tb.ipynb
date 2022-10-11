import os
import sys

import tensorflow as tf

def load_graph(graph_pb_path):
  with open(graph_pb_path,'rb') as f:
    content = f.read()
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(content)
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')
  return graph

  
def graph_to_tensorboard(graph, out_dir):
  with tf.Session():
    train_writer = tf.summary.FileWriter(out_dir)
    train_writer.add_graph(graph)
  
  
def main(out_dir, graph_pb_path):
  graph = load_graph(graph_pb_path)
  graph_to_tensorboard(graph, out_dir)
  
if __name__ == "__main__":
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  main(*sys.argv[1:])
