import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten, Layer
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

np.random.seed(3)

classes = 3

X_train = np.loadtxt('data/X_train.txt')
Y_train = np.loadtxt('data/y_train.txt')
X_test = np.loadtxt('data/X_test.txt')
Y_test = np.loadtxt('data/y_test.txt')

Y_train = keras.utils.to_categorical(Y_train-1, classes)
Y_test = keras.utils.to_categorical(Y_test-1, classes)

def argh(x):
    print("x is", x, K.cast(K.argmax(x), dtype='float32'))
    return K.cast(K.argmax(x, axis=0), dtype='float32')


model = Sequential()
model.add(Dense(40, input_dim=186, activation='relu', name='x'))     # take X features number from create-testset.js here!
# model.add(Dropout(0.5, seed=5, name='dropout1'))
# model.add(Dense(20, activation='relu', name='hidden4'))
model.add(Dense(classes, activation='softmax'))

# model.add(Dense(25, activation='relu', name='hidden3'))

# model.add(Dense(100, activation='relu', name='hidden1'))
# model.add(Dense(50, activation='relu', name='hidden2'))
# model.add(Dropout(0.5, seed=5, name='dropout1'))
# # model.add(Dense(10, activation='relu', name='hidden4'))
# model.add(Dense(5, activation='relu', name='hidden5'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=50, epochs=100, validation_data=(X_test, Y_test))
model.add(Lambda(lambda x: K.cast(K.argmax(x, axis=0), dtype='float32'), name='y_pred'))
model.save('data/trained.h5')

# Convert into TensorFlow PB file (as uTensor needs this)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.io.write_graph(frozen_graph, 'data', 'trained.pb', as_text=False)

# Create tensorboard log directory

with tf.Session() as sess:
    model_filename ='data/trained.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:

        data = compat.as_bytes(f.read())
        sm = saved_model_pb2.SavedModel()
        sm.ParseFromString(data)
        #print(sm)
        if 1 != len(sm.meta_graphs):
            print('More than one graph found. Not sure which to write')
            sys.exit(1)

        g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)

LOGDIR='tensorflow-logs'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
train_writer.flush()
train_writer.close()
