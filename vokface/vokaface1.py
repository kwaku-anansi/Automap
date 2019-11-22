from os import listdir
from os.path import isfile, join, splitext
import face_recognition
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
import pickle

faces_dict = {}




app = Flask(__name__)
CORS(app)


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
    faces = face_recognition.face_encodings(loaded_image)


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


    uploaded_faces = face_recognition.face_encodings(img)

    faces_found = len(uploaded_faces)
    faces = []

    if faces_found:
        face_encodings = list(faces_dict.values())
        for uploaded_face in uploaded_faces:
            match_results = face_recognition.compare_faces(
                face_encodings, uploaded_face, tolerance=0.4)
            for idx, match in enumerate(match_results):
                if match:
                    match = list(faces_dict.keys())[idx]
                    match_encoding = face_encodings[idx]
                    dist = face_recognition.face_distance([match_encoding],
                            uploaded_face)
                            
                    match1 = np.array(match)
                    mmatch = np.argmin(match)
                            
                    faces.append(result)
    
    return (faces)
    
        
    



@app.route('/vokaface/api/identify', methods=['POST'])
def web_recognize():

    pickle_in = open("dict.pickle","rb")
    faces_dict = pickle.load(pickle_in)
    
    file = extract_image(request)

    if file and is_picture(file.filename):
      
        return jsonify(detect_faces_in_image(file))
    else:
        raise BadRequest("Given file is invalid!")


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
    app.run(host='0.0.0.0', port=8080, debug=False)
