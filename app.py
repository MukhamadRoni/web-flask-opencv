from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np

app=Flask(__name__)
camera = cv2.VideoCapture(0)

roni_image = face_recognition.load_image_file('roni_img/1.jpg')
roni_image_encoding  = face_recognition.face_encodings(roni_image)[0]

john_image = face_recognition.load_image_file('jm_img/jm.jpg')
john_image_encoding = face_recognition.face_encodings(john_image)[0]

known_face_encodings = [
    roni_image_encoding,
    john_image_encoding
]
known_face_names = [
    'Roni',
    'John'
]

face_locations = []
face_encodings = []
face_names = []
process_this_frames = True

def generate_frames():
    while True:
        ## Read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = 'Unknown'

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                    
            ret, buffer=cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)