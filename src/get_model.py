from keras.layers import Layer, Flatten, Conv1D, Activation, GlobalAveragePooling1D, Reshape, merge, Input, BatchNormalization, GRU, concatenate, Dense, Dropout, Flatten
from keras.models import Model
from keras import regularizers
from tensorflow.keras.optimizers import Adam

# the best accuracy: 77%
def get_cnn_rnn_model():
  left_hidden_units = 64

  _input = Input(shape=(10, 2))
  right_model_input = Conv1D(128, 3, padding="same")(_input)
  right_model_input = BatchNormalization()(right_model_input)
  right_model_input = Activation("relu")(right_model_input)
  right_model_input = Conv1D(256, 3)(right_model_input)
  right_model_input = BatchNormalization()(right_model_input)
  right_model_input = Activation("relu")(right_model_input)
  right_model_input = Conv1D(128, 3)(right_model_input)
  right_model_input = BatchNormalization()(right_model_input)
  right_model_input = Activation("relu")(right_model_input)
  right_model_input = GlobalAveragePooling1D()(right_model_input)

  left_model_input = GRU(
    units=left_hidden_units, activation="relu", kernel_initializer="orthogonal", recurrent_initializer="orthogonal",
    bias_initializer="zeros", kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
    bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,
    return_state=False, go_backwards=False, stateful=False, unroll=False)(_input)

  left_model_input = GRU(
    units=left_hidden_units, activation="relu", kernel_initializer="orthogonal", recurrent_initializer="orthogonal",
    bias_initializer="zeros", kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
    bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,
    return_state=False, go_backwards=False, stateful=False, unroll=False)(left_model_input)
  
  left_model_input = Flatten()(left_model_input)

  model_input = concatenate([right_model_input, left_model_input], axis=1)

  model_input = Dropout(0.5)(model_input)
  model_input = Dense(128)(model_input)
  model_input = BatchNormalization()(model_input)
  model_input = Dense(2)(model_input)
  model_input = BatchNormalization()(model_input)
  model_output = Activation("sigmoid")(model_input)

  model = Model(_input, model_output)

  model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

  return "CNN-RNN", model

# average accuracy: 66%
def get_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(128, 3, padding="same", input_shape=input_shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv1D(256, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv1D(128, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation("sigmoid"))
    
    model.compile(loss="binary_crossentropy",
                optimizer="adam") 
    return "CNN", model