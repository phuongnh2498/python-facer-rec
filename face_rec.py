import base64
import face_recognition as fr
import os
import cv2
import numpy as np
import json
from time import sleep

TOLERANCE = 0.6
TRAIN_ALL_FOLDER = "./model_faces/ALL/"


def get_encoded_face(image_file=None):
    face = fr.load_image_file(image_file)
    encoding = fr.face_encodings(face)[0]
    return encoding


def get_encoded_faces(classID="ALL"):
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}
    pathToGetEncoded = "./model_faces/"+classID+"/"
    if(not os.path.exists(pathToGetEncoded)):
        pathToGetEncoded = TRAIN_ALL_FOLDER

    for dirpath, dnames, fnames in os.walk(pathToGetEncoded):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                print("loading image file...")
                face = fr.load_image_file(pathToGetEncoded + f)
                print("encoding image file...")
                encoding = fr.face_encodings(face)[0]
                print("Finish 1 image file...")
                encoded[f.split(".")[0]] = encoding
    print(encoded)
    return encoded


def check_unknown_image_encoded(im=None):
    print("im")
    print(im)
    if im == []:
        im = []
    img = cv2.imdecode(im, 1)
    encoding = True if len(fr.face_locations(img)) > 0 else False

    return encoding


def classify_face(im, tolerance=0.6, faces_model={}):
    print("decoding...")
    img = cv2.imdecode(im,  1)
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    print("getting faces...")
    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(
        img, face_locations)

    print("getting encoded faces...")

    faces = faces_model
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = fr.compare_faces(
            faces_encoded, face_encoding, tolerance=tolerance)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = fr.face_distance(
            faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
    json_data = getRecognitionData(face_locations, face_names)
    print(json_data)
    return json_data


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "--"+str(counter) + extension
        counter += 1

    return path


def getRecognitionData(face_locations=list([]), face_names=list([])):
    dataArr = []
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        tempData = dict({'cordinate': {'top': top, 'right': right,
                        'bottom': bottom, 'left': left}, 'name': name})
        dataArr.append(tempData)
    return json.dumps(dataArr, indent=1)
