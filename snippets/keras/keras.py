from keras.models import Sequential

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# Create a sequential model
model = Sequential()

# Add a fully conected layer with n units
# ATENTION: input_dim must be set only to the input layer
model.add(Dense(n, input_dim=X.shape[1]))

# Add an activation layer: 
# 'softmax', 'relu', 'tanh', 'softmax' and others
model.add(Activation('softmax'))

# The previous two lines of code can be written as:
mode.add(Dense(n, activation='softmax', input_dim=X.shape[1]))

# Add as many layers as needed, including the output layer 
# with its activation function

# Compile the code, choosing loss function, optimizer (sgd, adam, RMSProp...) and metrics (accuracy, mae, mse...)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])

# Display model architecture:
model.summary()

# Train the model, choosing the number of epochs and verbosity
model.fit(X, y, epochs=1000, verbose=0)

# Evaluate model metrics
model.evaluate(X_val,y_val)

# Predict using new data
model.perdict_proba(X_test)