from utils.functions import make_folders, frame_colection, make_variables
import os 
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

DATA_PATH = os.path.join('./data/processed_data/MP_Data') 
actions = ['hola', 'a', 'b', 'c', 'i', 'n', 'bien']
no_sequences = 16
labels_dict = {label:num for num, label in enumerate(actions)}

# create variables to use for training - saved as sequences and labels but these are essentially X and y
sequences, labels = make_variables(DATA_PATH, no_sequences, labels_dict, actions)
# save as x and y variables 
X = np.array(sequences)
y = to_categorical(labels).astype(int)

## allows to monitor accuracy while its training
# check on terminal- into Logs folder:  tensorboard --logdir=. 
# copy link 
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
# Long short-term memory 
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10,126))) # returning sequences for next layer 
model.add(LSTM(128, return_sequences=True, activation='relu')) # returning sequences for next layer 
model.add(LSTM(64, return_sequences=False, activation='relu'))
# Dense layer: each neuron receives input from all the neurons of previous layer
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

# loss categorical_crossentropy because it is a multiclass classification
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X, y, steps_per_epoch = 10, epochs=200, callbacks=[tb_callback])

model.save('my_model_manos16')