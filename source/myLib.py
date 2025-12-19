# Synthos

import cv2

def detectHands(hands, frame):
	imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the BGR image to RGB.
	results = hands.process(imgRGB) # hands detection
	return results

def drawHands(results, frame, mp_hands, mp_drawing):
	if results.multi_hand_landmarks:
	#print(len(results.multi_hand_landmarks))
		for handLms in results.multi_hand_landmarks:
			for id, lm in enumerate(handLms.landmark):
					h, w, c = frame.shape
					cx, cy = int(lm.x * w), int(lm.y * h)
					# Draw
					if id == 4 :
						cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
					if id == 8 :
						cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
					if id == 12 :
						cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
					if id == 16 :
						cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
					if id == 20 :
						cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
			mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

def drawParams(parameters, frame, debug_mode=False):
	y = 40
	for param in parameters.items():
		if debug_mode or param[0] in ("note_min"):
			frame = cv2.putText(frame, str(param), (10,y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
			y += 40

def map_range(x, in_min, in_max, out_min, out_max):
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

scales = {'chromatic':(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
		  'pentatonic':(0, 2, 4, 7, 9),
		  'major':(0, 2, 4, 5, 7, 9, 11),
		  'natural minor':(0, 2, 3, 5, 7, 8, 10),
		  'harmonic minor':(0, 2, 3, 5, 7, 8, 11)}

def getNoteInScale(value, scale):
	s = scales[scale]
	return value//len(s)*12 + s[value%len(s)]

def playNotes(parameters, results):
	parameters['new_notes'].clear()
	if results.multi_hand_landmarks:
	#print(len(results.multi_hand_landmarks))
		for handLms in results.multi_hand_landmarks:
			for id, lm in enumerate(handLms.landmark):
				if id == 8 : # INDEX_FINGER_TIP
					x = int(map_range(lm.x, 0, 1, 0, len(scales[parameters['scale']])))
					y = int(map_range(lm.y, 0, 1, parameters['octaves'], 0))
					new_note = x + y*len(scales[parameters['scale']])
					new_note = getNoteInScale(new_note, parameters['scale']) + parameters['note_min']
					parameters['new_notes'].add(new_note)

def getNoteName(midi):
	return ('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')[midi%12]

def drawScale(parameters, frame):
	h, w, c = frame.shape
	X = len(scales[parameters['scale']])
	Y = parameters['octaves']
	for y in range(Y):
		for x in range(X):
			frame = cv2.rectangle(frame, ((x*w)//X, (y*h)//Y), (((x+1)*w)//X, ((y+1)*h)//Y), (255,0,0), 2, cv2.LINE_AA)
			note = getNoteInScale(x, parameters['scale']) + parameters['note_min']
			frame = cv2.putText(frame, getNoteName(note), (int((x+0.3)*w//X), int((y+0.95)*h//Y)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)