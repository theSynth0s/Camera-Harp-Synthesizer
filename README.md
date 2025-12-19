# Camera-Harp-Synthesizer

## Introduction

Play music touch-free by simply moving your hands in front of your camera.
This project is an interactive instrument that transforms hand movements into sound in real-time.
The program detects your hand position via webcam to trigger notes on a virtual grid inside a choosen musical scale.

## Results and  accuracy

As soon as a hand is detected in the camera's field of view, the corresponding note is played.
The software is relatively good at detecting hands even in bright or low light conditions.
Detection also works if the hand is partially covered or partially outside the field of view.

![flathand](/doc/flathand.png)
![outhand](/doc/outhand.png)
![coveredhand](/doc/coveredhand.png)

## Improvement perspectives

## Dependencies

The program was made in Python with the help of some libraries.
It use **Tkinter** (the python integrated lib) for the interface.
It also use **OpenCV** for managing camera capture and display preview.
And **MediaPipe** for hand recognition and tracking.

Install dependencies with : `pip install -r requirements.txt`

There is a lot of compatibility issues with **MediaPipe**.
You should install `mediapipe==0.10.21` and work with python version 12 or less (13 and 14 are not compatible yet).

## More

This is more a proof of concept than a finiched project so feel free to make a pull requests if you found a bug or if you want to contribute.

## Sources

- https://trymypy.com/python-synthesizer-build-your-own-sound-generator-with-pyaudio-and-numpy/
- https://medium.com/@kyang3200/deeplearning-hands-tracking-by-mediapipe-b91b5bf252e8
- https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md
- https://github.com/Sousannah/hand-tracking-using-mediapipe/tree/main
- https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- https://diymidicontroller.com/midi-note-chart/
