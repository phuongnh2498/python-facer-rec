from http import client
from numpy import record
from pymongo import MongoClient

DATABASE_NAME = "encoded_face_db"
TABLE_NAME = "model_faces_records"
PASSWORD = "YTC2ROxSyhuRQz2E"
client = MongoClient('mongodb+srv://somedatabase:'+PASSWORD +
                     '@cluster0.fcsig.mongodb.net/'+DATABASE_NAME+'?retryWrites=true&w=majority')
db = client.get_database(DATABASE_NAME)
face_rec_table = db.model_faces_records

new_model_image = {
    'face_name': 'Phuong_PRO1',
    'base64img': 'abc123',
    'encoded': [123.321, 213.032]
}
new_model_image2 = {
    'face_name': 'Phuong_PRO2',
    'base64img': 'abc123',
    'encoded': [123.321, 213.032]
}


def getFaces():
    print(face_rec_table.count_documents({}))
    return []


def addFaces():
    print(face_rec_table.count_documents({}))
    face_rec_table.insert_many([new_model_image, new_model_image2])
    return []


addFaces()
