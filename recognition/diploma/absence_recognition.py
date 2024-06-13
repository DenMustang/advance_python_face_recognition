import cv2
import face_recognition
from datetime import date
import numpy as np
import os

known_students = {
    "Anna Koshenko": "faces/anna_koshenko.jpg"
}

def recognize_faces():
    known_face_encodings = []
    known_face_names = []
    for name, image_file in known_students.items():
        student_image = face_recognition.load_image_file(image_file)
        student_face_encoding = face_recognition.face_encodings(student_image)[0]
        known_face_encodings.append(student_face_encoding)
        known_face_names.append(name)

    face_locations = []
    face_encodings = []
    face_names = []
    presence_list = []

    # fyi: it initialises in the slot 1, if it changes just return it back to slot 0
    video_capture = cv2.VideoCapture(0)

    presence_file = "presence.txt"
    if os.path.isfile(presence_file):
        with open(presence_file, "r") as file:
            presence_data = file.readlines()
    else:
        presence_data = []

    while True:
        ret, frame = video_capture.read()
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 160)

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)


            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

            if name != "Unknown" and name not in presence_list:
                presence_list.append(name)
                today = date.today().strftime("%d/%m/%Y")
                with open(presence_file, "a") as file:
                    file.write(f"{today} - {name}\n")

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            accuracy = (1 - face_distances[best_match_index]) * 100
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, f"{name} ({accuracy:.2f}%)", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    absence_file = "absence.txt"
    with open(absence_file, "w") as file:
        for student_name in known_students.keys():
            if student_name not in presence_list:
                today = date.today().strftime("%d/%m/%Y")
                file.write(f"{today} - {student_name}\n")

    while cv2.waitKey(1) & 0xFF == ord('q'):
        break
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    recognize_faces()
