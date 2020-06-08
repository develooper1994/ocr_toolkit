##Base classes
class ImageIOErrors(Exception):
    pass


class ImageOperationErrors(Exception):
    pass


class ModelLoadError(Exception):
    """
    Raise when model unable to read
    """
    pass


class CreateObjectError(Exception):
    """
    Raise when creating object/instance failed
    """
    pass


class OutOfBoundsError(Exception):
    # def __init__(self):
    #     super().__init__(self,"Bounding boxes are out of bounds.")
    """
    Raise if bounding boxes are not inside of image
    """
    pass


class WrongModelTypeError(Exception):
    """
    Raise when model type is not alpr or mobilenet during creating LicencePlateRecognition object
    """
    pass


class GpuExistsError(Exception):
    """
    Raise if gpu usage true but there is no gpu
    """
    pass


class ArrayTypeError(Exception):
    pass


##Child exceptions
class ReadImageError(ImageIOErrors):
    """
    Raise when image unable to read
    """
    pass


class WriteImageError(ImageIOErrors):
    """
    Raise when image unable to write
    """
    pass


class ImageChannelError(ImageOperationErrors):
    """
    Raise when Image is not RGB
    """
    pass


class NotNumpyArrayError(ArrayTypeError):
    """
    Raise when array is not numpy array
    """
    pass
