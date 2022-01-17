import base64
import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
import json
from time import sleep

TOLERANCE = 0.6
TRAIN_ALL_FOLDER = "./model_faces/ALL/"


def get_encoded_faces(classID=""):
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    pathToGetEncoded = TRAIN_ALL_FOLDER
    encoded = {}
    if(classID):
        pathToGetEncoded = "./model_faces/"+classID+"/"
        if(not os.path.exists(pathToGetEncoded)):
            pathToGetEncoded = TRAIN_ALL_FOLDER

    for dirpath, dnames, fnames in os.walk(pathToGetEncoded):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file(pathToGetEncoded + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
    return encoded


def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_face(im, classID=""):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces(classID)
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    # img = cv2.imread(im, 1)
    img = cv2.imdecode(im, 1)
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]

    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(
        img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(
            faces_encoded, face_encoding, tolerance=TOLERANCE)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(
            faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20),
                          (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom - 15),
                          (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left - 20, bottom + 15),
                        font, 1.0, (255, 255, 255), 2)
    # convert data to json
    json_data = getRecognitionData(face_locations, face_names)
    # print(json_data)
    # return json_data
    # Display the resulting image
    # while True:
    #     cv2.imshow('Video', img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         return  json_data
    return json_data


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "--"+str(counter) + extension
        counter += 1

    return path


def deleteFromTrainFolder(classID="", userID="", imgName=""):
    count = 0
    try:
        folder = TRAIN_ALL_FOLDER
        if classID:
            folder = "./model_faces/"+classID
        if(not os.path.exists(folder)):
            return count
        if imgName:
            os.remove(os.path.join(folder, imgName))
            return 1
        for subdir, dirs, files in os.walk(folder):
            for file in files:
                if(userID in file):
                    count += 1
                    os.remove(os.path.join(subdir, file))
        return count
    except:
        print("some thing gone wrong")
    return 0


def getImgFromTrainFolder(classID="", userID=""):
    folder = "./model_faces/"+classID
    imgList = []
    if(not os.path.exists(folder)):
        return json.dumps(imgList)
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if(userID in file):
                with open(os.path.join(subdir, file), "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                    tempData = dict(
                        {"imgName": str(file), "imgbase64": str(encoded_string)})
                    imgList.append(tempData)
    return json.dumps(imgList, indent=1)


def getRecognitionData(face_locations=list([]), face_names=list([])):
    dataArr = []
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        tempData = dict({'cordinate': {'top': top, 'right': right,
                        'bottom': bottom, 'left': left}, 'name': name})
        dataArr.append(tempData)
    return json.dumps(dataArr, indent=1)
