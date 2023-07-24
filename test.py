import cv2
from deepface import DeepFace
from collections import defaultdict

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(1)

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Initialize emotion count dictionary for each frame
    emotion_counts = defaultdict(int)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x - 20, y - 60), (x + w + 15, y + h + 80), (0, 0, 255), 2)
        face = frame[y:y + h, x:x + w]  # Crop face region for individual face analysis
        try:
            results = DeepFace.analyze(face, actions=['emotion'])
            dominant_emotion = results[0]['dominant_emotion']

            # Increment count for the dominant emotion
            emotion_counts[dominant_emotion] += 1

            # Text in the top-left corner
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (x - 15, y + h + 25), cv2.FONT_HERSHEY_PLAIN,
                        0.8, (255, 0, 0), 1,
                        cv2.LINE_AA)
        except:
            print("No face detected")

    # Generate the emotion count dictionary
    emotion_count_dict = {emotion: count for emotion, count in emotion_counts.items()}

    # Print and display the emotion counts
    for emotion, count in emotion_counts.items():
        print(f"{emotion}: {count}")



    # Resize the frame to match the desired window size
    window_width, window_height = int(frame.shape[1] ), int(frame.shape[0] )
    frame_resized = cv2.resize(frame, (window_width, window_height))
    # Generate the bottom text with emotion counts
    text_bottom = "Emotion Counts: " + ", ".join(
        [f"{emotion}: {count}" for emotion, count in emotion_count_dict.items()])
    cv2.putText(frame_resized, text_bottom, (10, frame_resized.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0),
                2, cv2.LINE_AA)
    cv2.imshow('video', frame_resized)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
