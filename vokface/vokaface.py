from os import listdir
from os.path import isfile, join, splitext
import face_recognition
from flask import Flask, jsonify, render_template, request, Response
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
import pickle
import numpy as np
import cv2

faces_dict = {}




app = Flask(__name__)
CORS(app)

video = cv2.VideoCapture(0)


def is_picture(filename):
    image_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in image_extensions


def get_all_picture_files(path):
    files_in_dir = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return [f for f in files_in_dir if is_picture(f)]


def remove_file_ext(filename):
    return splitext(filename.rsplit('/', 1)[-1])[0]

def calc_face_encoding(image):
    
    loaded_image = face_recognition.load_image_file(image)
    fac_locations = face_recognition.face_locations(loaded_image, number_of_times_to_upsample=1, model="cnn")
    faces = face_recognition.face_encodings(loaded_image, fac_locations) 
    

    if len(faces) > 1:
        raise Exception(
            "Found more than one face in the given training image.")


    if not faces:
        raise Exception("Could not find any face in the given training image.")

    return faces[0]


def get_faces_dict(path):
    image_files = get_all_picture_files(path)
    return dict([(remove_file_ext(image), calc_face_encoding(image))
                 for image in image_files])
                 
                 
                 


def detect_faces_in_image(file_stream):

    img = face_recognition.load_image_file(file_stream)
    face_locations = face_recognition.face_locations(img)
    uploaded_faces = face_recognition.face_encodings(img, face_locations)

    faces_found = len(uploaded_faces)
    faces = []
    nmatch = 'No_Match'
    if faces_found:
        face_encodings = list(faces_dict.values())
        for uploaded_face in uploaded_faces:
            match_results = face_recognition.compare_faces(
                face_encodings, uploaded_face, tolerance=0.33)

            face_distances = face_recognition.face_distance(face_encodings, uploaded_face)    
            best_match_index = np.argmin(face_distances)
            if match_results[best_match_index]:
                name = list(faces_dict.keys())[best_match_index]
            else:
                name = nmatch   

            

    return (name)
  



@app.route('/vokaface/api/identify', methods=['POST'])
def web_recognize():

    pickle_in = open("dict.pickle","rb")
    faces_dict = pickle.load(pickle_in)
    
    file = extract_image(request)

    if file and is_picture(file.filename):
      
        return jsonify(detect_faces_in_image(file))
    else:
        raise BadRequest("Given file is invalid!")

@app.route('/video')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    """Video streaming generator function."""
    #pickle_in = open("dict.pickle","rb")
    #faces_dict = pickle.load(pickle_in)
    while True:
        rval, frame = video.read()
        rgb_frame = frame[:, :, ::-1]
        
       
        face_locations = face_recognition.face_locations(rgb_frame)
        uploaded_faces = face_recognition.face_encodings(rgb_frame, face_locations)
        
        faces_found = len(uploaded_faces)
        nmatch = 'No_Match'
        if faces_found:
            face_encodings = list(faces_dict.values())
            for uploaded_face in uploaded_faces:
                match_results = face_recognition.compare_faces(
                    face_encodings, uploaded_face, tolerance=0.40)

                face_distances = face_recognition.face_distance(face_encodings, uploaded_face)    
                best_match_index = np.argmin(face_distances)
                if match_results[best_match_index]:
                    name = list(faces_dict.keys())[best_match_index]
                else:
                    name = nmatch   

            

        #return (name)   

        for (top, right, bottom, left), uploaded_faces in zip(face_locations, uploaded_faces):    
        # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        cv2.imwrite('t.jpg', frame)
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
@app.route('/vokaface/api/faces', methods=['GET', 'POST', 'DELETE'])
def web_faces():
    pickle_out = open("dict.pickle","wb")
    pickle.dump(faces_dict, pickle_out)
    pickle_out.close()

    if request.method == 'GET':
        return jsonify(list(faces_dict.keys()))

    file = extract_image(request)
    if 'id' not in request.args:
        raise BadRequest("Vokaface ID was not given!")

    if request.method == 'POST':
        try:
            new_encoding = calc_face_encoding(file)
            faces_dict.update({request.args.get('id'): new_encoding})
        except Exception as exception:
            raise BadRequest(exception)

    elif request.method == 'DELETE':
        faces_dict.pop(request.args.get('id'))

    return jsonify(list(faces_dict.keys()))


def extract_image(request):
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")

    file = request.files['file']
    if file.filename == '':
        raise BadRequest("Given file is invalid")

    return file


if __name__ == "__main__":
    print("Starting by generating encodings for found images...")

    faces_dict = get_faces_dict("faces")

    print("Starting Vokaface WebServer...")
    app.run(host='0.0.0.0', port=8081, debug=False)
    

    
    
    
    
