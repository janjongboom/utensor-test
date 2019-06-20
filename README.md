# uTensor test

So this is a trained model (see trained.pb), it has a few layers:

```py
model = Sequential()
model.add(Dense(40, input_dim=186, activation='relu', name='x'))     # take X features number from create-testset.js here!
model.add(Dropout(0.5, seed=5, name='dropout1'))
model.add(Dense(25, activation='relu', name='hidden3'))
model.add(Dense(5, activation='relu', name='hidden5'))
model.add(Dense(classes, activation='softmax', name='y_pred'))
```

All input data was normalized between -127 and 127.
