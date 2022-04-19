
from utils.functions import prob_viz, import_solutions, extract_keypoints, load_model, make_variables
import cv2 
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mp_hands, mp_drawing, mp_drawing_styles = import_solutions()

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 
# Actions that we try to detect
actions = ['a', 'b', 'c', 'i', 'n']
key = [2,3,4,5,6]
key_action = dict(zip(key,actions))
# number of videos collected 
no_sequences = 13
# Videos are going to be 20 frames in length
sequence_length = 10

# sequences, labels = make_variables(DATA_PATH, no_sequences, labels_dict, actions)
model = load_model('./model/video_model_weights_11')

sequences = []
window = []
sentence = []
threshold = 0.1
colors = [(245,117,16), (117,245,16), (16,117,245)]

## Code to visualize predictions
# Use abcin9.mp4, 
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():

        # Capture frame-by-frame (reads video feed)
        # returns boolean (read properly?) and image 
        success, image = cap.read()
        
        # if image isn't read properly, the window closes
        if not success:
            cap.release()
            cv2.destroyAllWindows()
        # if the image is read properly, we continue with steps
        else:

            # gives option to exit video whenever you want 
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            # mediapipe detection: make prediction (process) then make image writeable
            # to draw hand annorations on the image
            image.flags.writeable = False
            image = image
            cv2.imshow('frame', cv2.flip(image, 1))
            results = hands.process(image)
            image.flags.writeable = True

            # draw the keypoint landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

            frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            # Prediction
            keypoints = extract_keypoints(results, frameWidth, frameHeight)
            sequences.append(keypoints)
            if len(sequences) > 10:
                sequence = sequences[-10:]
                try:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                except:
                    print('')

                try:
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])
                except:
                    print('')

                if len(sentence) > 5: 
                    sentence = sentence[-5:]
            
                # Viz probabilities
                try:
                    image = prob_viz(res, actions, image, colors)
                except:
                    print('')
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

    cap.release()
    cv2.destroyAllWindows()