import train as t
import cv2
import face_recognition
import numpy as np
import keyboard

t.train()
cam_id = 0
cap = cv2.VideoCapture(cam_id)
while True:
    ret, im = cap.read()

    face_locations = face_recognition.face_locations(im)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        img = im[top:bottom, left:right]
        img = cv2.resize(img, (48, 48))
        img = img.astype(np.float64) / 255

        predict_x = t.model.predict(img[np.newaxis, :, :, :])
        classes_x = np.argmax(predict_x, axis=1)

        cv2.rectangle(im, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(im, t.labels[classes_x[0]], (left + 2, bottom + 6), font, 0.4, (255, 255, 255), 1)

    if keyboard.is_pressed('q'):
        break
    cv2.imshow("face", im)
    cv2.waitKey(1)
cap.release()
