from utils.functions import make_folders, frame_colection, make_variables
import os 

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('./data/processed_data/MP_Data') 
# Actions that we try to detect
actions = ['hola', 'a', 'b', 'c', 'i', 'n']
key = [1,2,3,4,5,6,7]
key_action = dict(zip(key,actions))
# Thirty videos worth of data
no_sequences = 16
# Videos are going to be 10 frames in length
sequence_length = 10

make_folders(actions, no_sequences, DATA_PATH)

# key_to press = according to key asociated to action, count
# inputs: frame_colection(key_action (dict), key_to_press, count, input=0):
frame_colection(key_action, 1, 14, input=0)
