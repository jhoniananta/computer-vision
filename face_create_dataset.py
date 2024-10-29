import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
dataset_path = "dataset/"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

person_id = 1
count = 0
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30, 30))

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
        count += 1
        cv2.imwrite(dataset_path + "person-" + str(person_id) + "-" + str(count) + ".jpg", gray[y:y+height, x:x+width])
    
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        break
    elif count == 500:
        break

cap.release()
cv2.destroyAllWindows()