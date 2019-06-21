# uTensor test

So this is a trained model (see trained.pb), it has a few layers:

```py
model = Sequential()
model.add(Dense(40, input_dim=186, activation='relu', name='x'))     # take X features number from create-testset.js here!
model.add(Dropout(0.5, seed=5, name='dropout1'))
model.add(Dense(25, activation='relu', name='hidden3'))
model.add(Dense(5, activation='relu', name='hidden5'))
model.add(Dense(3, activation='softmax', name='y_pred'))
```

All input data was normalized between -127 and 127.

There are two issues with this code:

1. Prediction is wrong (or I don't know how to read the prediction). TensorFlow returns 3 values for every prediction (e.g. `[ 0.0012, 0.9888, 0.0231 ]`), which map to classes `1`, `2`, and `3` (all y values in my dataset are one of these). I've included 3 samples from my dataset which are classified by TensorFlow as `1`, `2`, and `3`. uTensor classifies them as:
    * `0.999809`.
    * `0.000898`.
    * `0.999809`.
    * Which looks more like probablity.
2. It allocates `186*40*4` bytes memory. It should only allocate `186*40` bytes memory.

## Example TF output

```
array([[9.9981266e-01, 1.8708695e-04, 2.8274363e-07],
       [9.9981266e-01, 1.8708695e-04, 2.8274363e-07],
       [9.9981266e-01, 1.8708695e-04, 2.8274363e-07],
       ...,
       [0.0000000e+00, 1.3602355e-10, 1.0000000e+00],
       [0.0000000e+00, 2.2191929e-12, 1.0000000e+00],
       [0.0000000e+00, 2.5003938e-12, 1.0000000e+00]], dtype=float32)
```

You can see this for the examples I included in `main.cpp` via:

```
python3 classify.py utensor-test.txt
```
