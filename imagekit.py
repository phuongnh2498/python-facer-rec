# SDK initialization

from imagekitio import ImageKit

imagekit = ImageKit(
    private_key='private_+MCQvPYe1/AsieTiSEtjkTkwx2I=',
    public_key='public_ZY/wD4V7pXh+jw/LIP776Nz1waM=',
    url_endpoint='https://ik.imagekit.io/th8f4o3b200'
)


def upload():
    upload = imagekit.upload_file(
        file=open("./model_faces/All/bill gates.jpg", "rb"),
        file_name="testing_upload_binary_signed_private.jpg",
        options={
            "response_fields": ["is_private_file", "tags"],
            "is_private_file": False,
            "folder": "/testing-python-folder/",
            "tags": ["abc", "def"]
        },
    )
    print(str(upload))


upload()
