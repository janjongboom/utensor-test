import tensorflow as tf
import numpy as np
import sys

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

# We use our "load_graph" function
graph = load_graph("./data/trained.pb")

# We can verify that we can access the list of operations in the graph
for op in graph.get_operations():
    print(op.name)     # <--- printing the operations snapshot below
    # prefix/Placeholder/inputs_placeholder
    # ...
    # prefix/Accuracy/predictions

# We access the input and output nodes
x = graph.get_tensor_by_name('prefix/x_input:0')
y = graph.get_tensor_by_name('prefix/y_pred/ArgMax:0')

# We launch a Session
with tf.Session(graph=graph) as sess:

    test_features = [
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 8, -17, 0, 0, 22, -6, 4, 75, -25, -30, 0, 0, 31, 0, 0, -31, 0, 0, 0, 0, 0, 0, 16, 0, 0, -16, 0, 29, 0, 16, -1, -71, 16, 0, 1, 26, -3, 70, 5, 0, 0, 0, -25, 0, 22, 0, 0, 9, 0, 0, 24, 18, 0, 8, 4, 0, 0, -5, 0, 32, 0, 0, 0, -18, 0, 32, 0, 0, 18, 0, -67, 13, 0, 1, 0, 0, -4, 17, 0, -2, -17, 0, 0, 0, 0, 2, 0, -74, 0, 0, -23, -7, 29, 0, -20, 67, 17, 19, -78, 6, -3, 1, 0, 0, 0, -6, -16, -3, -17, 0, -16, 17, 0, 0, 11, 0, -11, -1, 17, -21, 0, 0, 0, -4, -17, 0, 1, 0, 0, 7, 25, -8, 0, 72, -23, 0, 0, -8, -66, -70, 7, 0, 0, 0, 66, -4, 1, 17, 6, -6, 0, 68, -9, -17, -80, -16, 0, -17, 20, 23, 17, -21, 0, 0, 0, -23, 6, -13, 0, 2, -17, 21, 70, -16, -21, -65, -16, 0, 65, -7, 29, 17, -25],
       [78, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 17, 0, -22, -17, 0, 16, -10, 0, -82, 38, 0, 84, 69, 0, 0, 0, 0, -19, 0, 0, 0, 0, 0, 0, 18, -1, 0, -18, 1, -69, -80, 0, -2, 80, 0, 0, 0, 0, -3, 18, 0, 103, -18, 0, -3, 27, 0, -24, -97, 0, 0, -8, 0, 0, 78, 0, 0, 0, 0, 22, 0, 0, 9, 0, 0, -32, 18, 0, 0, -18, 0, 0, -76, 0, 0, 8, 0, 0, 0, 0, 19, 68, 0, -19, 0, 0, -78, -97, 0, 10, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 2, 0, 16, -25, -1, -16, -2, 1, 0, -5, 0, 23, -25, -1, 0, 0, 0, -23, -15, 1, -97, 15, 0, -32, 58, 0, 32, 71, 0, 24, -98, 0, 0, 0, 0, 0, 0, 0, 74, 22, 0, 0, 75, 0, -73, 0, 0, -4, 0, 0, 0, 0, 0, -20, 0, 0, 0, 0, 0, 0, 17, 0, 0, -17, 0, 0, 0, 0]
    ]
    # compute the predicted output for test_x
    pred_y = sess.run( y, feed_dict={x: test_features} )
    print('Begin output')
    print(pred_y)
    print('End output')
