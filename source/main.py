# Synthos

import tkinter as tk
from tkinter import ttk
from threading import *

import cv2
import mediapipe as mp 
# There is a lot of compatibility issues with mediapipe
# You should install mediapipe==0.10.21 and work with python version 12 or less, 13 and 14 are not compatible

from synthLib import *
import myLib

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

PREVIEW_SIZE = 1.6 # (0.5 - 4)
MODEL_COMPLEXITY = 0 # (0 or 1, 0 is faster and 1 is more accurate)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#cap.set(cv2.CAP_PROP_SETTINGS, 0) # display camera settings pop up (useful to get maximum fps)

def window_Tk(parameters):
	root = tk.Tk()
	try:
		style = ttk.Style()
		style.theme_use("clam")
	except Exception:
		pass
	app = SynthGUI(root, parameters)
	root.protocol("WM_DELETE_WINDOW", root.destroy)
	root.mainloop()
	print("thread 1 died")

def window_CV(parameters):
	cap = cv2.VideoCapture(0)

	with mp_hands.Hands(model_complexity=MODEL_COMPLEXITY, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=4) as hands:
		while cap.isOpened():
			ret, frame = cap.read()
			
			frame = cv2.flip(frame, 1)
			results = myLib.detectHands(hands, frame)
			frame = cv2.resize(frame, None, fx=PREVIEW_SIZE, fy=PREVIEW_SIZE)

			myLib.drawHands(results, frame, mp_hands, mp_drawing)
			myLib.drawParams(parameters, frame, debug_mode=False)
			myLib.drawScale(parameters, frame)
			cv2.imshow('Hand Synthesizer', frame)

			myLib.playNotes(parameters, results)

			if cv2.waitKey(5) & 0xFF == ord('q'): # EXIT with Q
				break

	cap.release()
	cv2.destroyAllWindows()
	print("thread 2 died")

def main():
	parameters = {}

	t1 = Thread(target=window_Tk, args=(parameters,))
	t2 = Thread(target=window_CV, args=(parameters,))

	t1.start()
	t2.start()


if __name__ == "__main__":
	main()