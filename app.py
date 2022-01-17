from pickle import TRUE
from xml.etree.ElementTree import tostring
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import base64
import numpy as np
import os
from flask_cors import CORS
from face_rec import classify_face, uniquify, deleteFromTrainFolder, getImgFromTrainFolder
# Post Folder
UPLOAD_FOLDER = 'model_faces/ALL'

# Init app

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)


@app.route('/', methods=['GET'])
def get():
    return jsonify({'msg': 'Hello World'})


@app.route('/delete-user-train-image', methods=['POST'])
def deleteImage():
    content = request.json
    count = 0
    if not 'userID' in content:
        return jsonify({'msg': "please enter userID"})
    # delete in all in all folder
    if not 'classID' in content:
        count = deleteFromTrainFolder(classID="", userID=content["userID"])
        return jsonify({'msg': "Deleted "+str(count)+" train Image from user " + content["userID"]})
    if 'imgName' in content:
        count = deleteFromTrainFolder(
            classID=content['classID'], userID=content["userID"], imgName=content['imgName'])
        return jsonify({'msg': "Deleted "+str(count)+" train Image from user " + content["userID"]})
    # clear all train of userID in class
    count = deleteFromTrainFolder(
        classID=content['classID'], userID=content["userID"])
    return jsonify({'msg': "successfully deleted " + str(count) + " train Image from user " + content["userID"]})


@app.route('/user-train-image', methods=['POST', 'GET'])
def ResolveTrainImage():
    # print(request.json)
    content = request.json
    if request.method == "POST":
        if('classID' in content):
            foldername = str(content['classID']).replace("/", "")
            app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER.split(
                "/")[0]+"/"+foldername
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
        imgdata = base64.b64decode(content['imgbase64'])
        path = uniquify(os.path.join(
            app.config['UPLOAD_FOLDER'], content['userID']+".jpg"))
        with open(path, 'wb') as f:
            f.write(imgdata)
        return jsonify({'msg': 'successfully added train image!'})
    if request.method == 'GET':
        jsonData = {}
        if not "userID" in content:
            jsonData = getImgFromTrainFolder(
                userID="donal numb", classID="IS1402")
        jsonData = getImgFromTrainFolder(
            userID=content["userID"], classID=content["classID"])
        return jsonData


@app.route('/recongize-user-image', methods=['POST'])
def processImage():
    # print(request.json)
    content = request.json
    b64img = base64.b64decode(content['imgbase64'])
    npimg = np.fromstring(b64img, dtype=np.uint8)
    if('classID' in content):
        return classify_face(npimg, classID=content['classID'])
    return classify_face(npimg)


# Run server
if(__name__ == '__main__'):
    app.run(debug=True)
