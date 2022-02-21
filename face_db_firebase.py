
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import numpy as np
import pickle

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

face_model_collection = db.collection('face_models_collection')


def addNewModel(encoded_face_arr=[], face_name="test--xyz", folder="test", image_url="test.com", image_id="abc123"):
    print("adding!")
    face_model = {
        "encoded_face": pickle.dumps(encoded_face_arr),
        "face_name": face_name,
        "folder": folder,
        "image_url": image_url,
        "image_id": image_id,
    }
    face_model_collection.add(face_model)
    print("done!")


def getModelALL():
    print("getting!")
    face_list = face_model_collection.get()
    for item in face_list:
        print(item.to_dict())


def getModelByFolder(folder="ALL"):
    print("getting face model...")
    face_dict = {}
    face_list = face_model_collection.where("folder", "==", folder).get()
    for item in face_list:
        model = item.to_dict()
        face_dict[model["face_name"]] = pickle.loads(model["encoded_face"])
    print(face_dict)
    return face_dict


def getModelByFolderForMobile(folder="ALL"):
    print("getting face model...")
    face_dict = []
    face_list = face_model_collection.where("folder", "==", folder).get()
    for item in face_list:
        model = item.to_dict()
        del model['encoded_face']
        face_dict.append(model)
    return face_dict


def deleteModelByImageID(image_id="abc123"):
    print("deleting face model...")
    doc_to_delete = face_model_collection.where(
        "image_id", "==", image_id).get()
    if len(doc_to_delete) <= 0:
        return False
    return face_model_collection.document(doc_to_delete[0].id).delete()
