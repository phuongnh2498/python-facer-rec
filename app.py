from pickle import TRUE
from flask import Flask, request, jsonify
import json
import numpy as np
import copy
import io
import PIL.Image as Image
import os
from flask_cors import CORS
from sqlalchemy import true
from face_db_firebase import addNewModel, deleteModelByImageID, getModelByFolder, getModelByFolderForMobile
from face_rec import classify_face, check_unknown_image_encoded, get_encoded_face, uniquify, deleteFromTrainFolder, getImgFromTrainFolder
from imagekit import deleteImageByID, uploadImage
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
    image_id = content['image_id']
    if not 'image_id' in content:
        return jsonify({'msg': "please enter image_id"})
    # delete image from firebase
    deleteModelByImageID(image_id=image_id)
    # delete image from imagekit
    deleteImageByID(image_id=image_id)
    return jsonify({'msg': "successfully deleted " + image_id})


@app.route('/user-train-image', methods=['POST', 'GET'])
def user_train_image():
    if request.method == "POST":
        content = dict(request.form)

        if not ('image_folder' in content):
            return jsonify({'msg': 'please enter image_folder'})
        if not 'image_file' in request.files:
            return jsonify({'msg': 'please select an image_file'})
        if not ('user_id' in content):
            return jsonify({'msg': 'please enter user_id'})

        image_file = request.files['image_file']
        user_id = str(content['user_id']).replace("/", "").upper()
        image_folder = str(content['image_folder']).replace("/", "").upper()
        # check if face in image
        npimg = np.frombuffer(image_file.read(), np.uint8)
        if not check_unknown_image_encoded(npimg):
            return jsonify({'msg': "can't detect face in image"})
        # upload imagekit
        res_imgkit = uploadImage(imageFile=io.BytesIO(npimg),
                                 imageName=user_id,
                                 folder=image_folder)
        # fetch response imagekit
        model_face_name = user_id+"--"+res_imgkit['fileId']
        model_image_url = res_imgkit['thumbnailUrl']
        # get encoded face arr
        encoded_arr = get_encoded_face(image_file=io.BytesIO(npimg))
        # post to firebase
        fire_res = addNewModel(encoded_face_arr=encoded_arr,
                               face_name=model_face_name,
                               folder=image_folder,
                               image_url=model_image_url,
                               image_id=res_imgkit['fileId']
                               )
        print(fire_res)
        return jsonify({'msg': 'successfully added train image! for user '+user_id})
    if request.method == 'GET':
        if request.args.get('user_id') is None:
            return jsonify({'msg': "please enter user_id or folder"})
        userID = request.args.get('user_id').upper()
        print("For user id" + userID)
        return json.dumps(getModelByFolderForMobile(folder=userID))


@ app.route('/recongize-user-image', methods=['POST'])
def processRecognizeImage():
    content = dict(request.form)
    tolerance = 0.6
    if not 'image_file' in request.files:
        return jsonify({'msg': 'wrong request format'})
    if not 'folder' in content:
        return jsonify({'msg': 'wrong request format'})
    if 'tolerance' in content:
        tolerance = float(content["tolerance"])

    # get faces from firebase
    face_dict = getModelByFolder(folder=content['folder'].upper())
    npimg = np.fromstring(request.files['image_file'].read(), np.uint8)

    return classify_face(npimg, tolerance=tolerance, faces_model=face_dict)


# Run server
if(__name__ == '__main__'):
    port = os.environ.get("PORT", 8000)
    app.run(debug=False, host='0.0.0.0', port=port)
