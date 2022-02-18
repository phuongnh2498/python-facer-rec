from pickle import TRUE
from xml.etree.ElementTree import tostring
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import base64
import numpy as np
import os
from flask_cors import CORS
from face_rec import classify_face, check_unknown_image_encoded, uniquify, deleteFromTrainFolder, getImgFromTrainFolder
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
        count = deleteFromTrainFolder(userID=content["userID"])
        return jsonify({'msg': "Deleted "+str(count)+" train Image from user " + content["userID"]})
    if 'imgName' in content:
        count = deleteFromTrainFolder(
            classID=content['classID'], imgName=content['imgName'])
        return jsonify({'msg': "Deleted "+str(count)+" train Image from user " + content["userID"]})
    # delete all train userID Images in class
    count = deleteFromTrainFolder(
        classID=content['classID'], userID=content["userID"])
    return jsonify({'msg': "successfully deleted " + str(count) + " train Image from user " + content["userID"]})


@app.route('/user-train-image', methods=['POST', 'GET'])
def ResolveTrainImage():
    # print(request.json)
    if request.method == "POST":
        content = dict(request.form)

        if not 'ImageFile' in request.files:
            return jsonify({'msg': 'please select an ImageFile'})
        if not ('userID' in content):
            return jsonify({'msg': 'please enter userID'})

        imgFile = request.files['ImageFile']
        npimg = np.fromstring(imgFile.read(), np.uint8)

        if not check_unknown_image_encoded(npimg):
            return jsonify({'msg': "can't detect face in image"})
        if 'classID' in content:
            # remove / from folder name
            foldername = str(content['classID']).replace("/", "")
            # concat folder path
            app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER.split(
                "/")[0]+"/"+foldername
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
        userID = str(content['userID']).replace("/", "")
        path = uniquify(os.path.join(
            app.config['UPLOAD_FOLDER'], userID+".jpg"))
        with open(path, 'wb') as f:
            f.write(npimg)
        return jsonify({'msg': 'successfully added train image! for user '+userID})
    if request.method == 'GET':
        jsonData = []
        userID = request.args.get('userID')
        classID = request.args.get('classID')
        if request.args.get('userID') is None and request.args.get('classID') is None:
            return jsonify({'msg': "please enter userID or classID"})
        elif request.args.get('classID') is None:
            jsonData = getImgFromTrainFolder(
                userID=str(userID))
        elif request.args.get('userID') is None:
            jsonData = getImgFromTrainFolder(
                classID=str(classID))
        else:
            jsonData = getImgFromTrainFolder(
                str(classID), str(userID))
        return jsonData


@app.route('/recongize-user-image', methods=['POST'])
def processRecognizeImage():
    content = dict(request.form)
    tolerance = 0.6
    npimg = np.fromstring(request.files['ImageFile'].read(), np.uint8)
    if('tolerance' in content):
        tolerance = float(content["tolerance"])
    if('classID' in content):
        return classify_face(npimg, classID=content['classID'], tolerance=tolerance)
    return jsonify({'msg': 'wrong base64 format'})


# Run server
if(__name__ == '__main__'):
    port = os.environ.get("PORT", 8000)
    app.run(debug=False, host='0.0.0.0', port=port)
