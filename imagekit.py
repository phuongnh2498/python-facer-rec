# SDK initialization

from imagekitio import ImageKit

imagekit = ImageKit(
    private_key='private_+MCQvPYe1/AsieTiSEtjkTkwx2I=',
    public_key='public_ZY/wD4V7pXh+jw/LIP776Nz1waM=',
    url_endpoint='https://ik.imagekit.io/th8f4o3b200'
)


def uploadImage(imageFile=None, imageName="image1", folder="ALL", tags=[]):
    print("upload image to imagekit")
    if imageFile == None:
        return {"msg": "sorry"}
    upload = imagekit.upload_file(
        file=imageFile,
        file_name=imageName+".jpg",
        options={
            "response_fields": ["is_private_file", "tags"],
            "is_private_file": False,
            "folder": "/"+folder+"/",
            "tags": tags
        },
    )
    response = upload['response']
    return response


def deleteImageByID(image_id="Tokuda"):
    return imagekit.delete_file(image_id)


# uploadImage()
# deleteImageByID("6212589baa2edf2a7118cf3b")
