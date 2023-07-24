import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(1)

while True:
    ret, frame = video.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x-20, y-60), (x + w+15, y + h + 80), (0, 0, 255), 2)
        face = frame[y:y + h, x:x + w]  # Crop face region for individual face analysis
        try:
            results = DeepFace.analyze(face, actions=['age', 'gender', 'race', 'emotion'])
            for result in results:
                age = result['age']
                gender = result['gender']
                race = result['race']
                emotion = result['emotion']

                dominant_gender, dominant_gender_value = max(gender.items(), key=lambda x: x[1])
                dominant_race, dominant_race_value = max(race.items(), key=lambda x: x[1])
                dominant_emotion, dominant_emotion_value = max(emotion.items(), key=lambda x: x[1])

                text_age = "Age: "+str(age)
                text_gender = "Gender: {} {:.2f}".format(dominant_gender, dominant_gender_value)
                text_race = "Race: {} {:.2f}".format(dominant_race, dominant_race_value)
                text_emotion = "Emotion: {} {:.2f}".format(dominant_emotion, dominant_emotion_value)

            #text in the bottom-left corner
                cv2.putText(frame, text_age, (x - 15, y + h + 25), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 1,
                            cv2.LINE_AA)
                cv2.putText(frame, text_gender, (x - 15, y + h + 40), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 1,
                            cv2.LINE_AA)
                cv2.putText(frame, text_race, (x - 15, y + h + 55), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 1,
                            cv2.LINE_AA)
                cv2.putText(frame, text_emotion, (x - 15, y + h + 70), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 1,
                            cv2.LINE_AA)

        except:
            print("no face")

    # Resize the frame to match the desired window size
    window_width, window_height = int(frame.shape[1]*1.8), int(frame.shape[0]*1.5)
    frame_resized = cv2.resize(frame, (window_width, window_height))
    cv2.imshow('video', frame_resized)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
