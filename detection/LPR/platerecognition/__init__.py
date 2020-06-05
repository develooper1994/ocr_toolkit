from os import path, environ

try:
    import ALPR
    import GpuExistsError, WrongModelTypeError
    import MobileNet
    import OCR
except:
    try:
        from alpr import ALPR
        from exceptions import GpuExistsError, WrongModelTypeError
        from mobilenet import MobileNet
        from ocr import OCR
    except:
        try:
            from platerecognition.alpr import ALPR
            from platerecognition.exceptions import GpuExistsError, WrongModelTypeError
            from platerecognition.mobilenet import MobileNet
            from platerecognition.ocr import OCR
        except:
            try:
                from LPR.platerecognition.alpr import ALPR
                from LPR.platerecognition.exceptions import GpuExistsError, WrongModelTypeError
                from LPR.platerecognition.mobilenet import MobileNet
                from LPR.platerecognition.ocr import OCR
            except:
                from detection.LPR.platerecognition.alpr import ALPR
                from detection.LPR.platerecognition.exceptions import GpuExistsError, WrongModelTypeError
                from detection.LPR.platerecognition.mobilenet import MobileNet
                from detection.LPR.platerecognition.ocr import OCR


def load_models():
    __work_dir = path.dirname(path.abspath(__file__)) + "/"

    if path.exists(__work_dir + "lpr-models/"):
        return

    from minio import Minio
    from minio.error import ResponseError

    __bucket_name = "ai-models"
    __prefix = "lpr-models/"
    __server = "192.168.1.11:9000" if "MINIO_SERVER" not in environ else environ['MINIO_SERVER']
    __access_key = "minio" if "MINIO_NAME" not in environ else environ['MINIO_NAME']
    __secret_key = "minio123" if "MINIO_SECRET" not in environ else environ['MINIO_SECRET']
    __secure = False

    # connect to minio
    try:
        minioClient = Minio(__server, access_key=__access_key, secret_key=__secret_key, secure=__secure)
        # get object list
        objects = minioClient.list_objects_v2(__bucket_name, prefix=__prefix, recursive=True)

        # get objects
        for obj in objects:
            minioClient.fget_object(__bucket_name, obj.object_name, __work_dir + obj.object_name)
    except ResponseError as e:
        print("cannot connect to minio server\n{}".format(e))
        quit(1)


class LicencePlateRecognition:
    def __init__(self, model_type="alpr", use_gpu=False, vehicle_thresh=0.35, plate_thresh=0.5):
        load_models()

        if model_type == "alpr":
            self.model = ALPR(use_gpu=use_gpu, vehicle_threshold=vehicle_thresh, plate_threshold=plate_thresh)
        elif model_type == "mobilenet":
            self.model = MobileNet()
        else:
            raise WrongModelTypeError


def get_ocr_model(use_gpu=False):
    load_models()
    return OCR(use_gpu=use_gpu)


##Test area
if __name__ == "__main__":
    load_models()
