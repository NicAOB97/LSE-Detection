from utils.functions import make_folders, frame_colection, make_variables
import os 
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('./data/processed_data/MP_Data') 
# Actions that we try to detect
actions = ['hola', 'a', 'b', 'c', 'i', 'n']
key = [1,2,3,4,5,6]
key_action = dict(zip(key,actions))
# Thirty videos worth of data
no_sequences = 14
# Videos are going to be 10 frames in length
sequence_length = 10

make_folders(actions, no_sequences, DATA_PATH)

# key_to press = according to key asociated to action, count
# frame_colection(key_action, 6, 13, 0 )

labels_dict = {label:num for num, label in enumerate(actions)}
sequences, labels = make_variables(DATA_PATH, no_sequences, labels_dict, actions)

# create x and y variables 
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

model.save('my_model')