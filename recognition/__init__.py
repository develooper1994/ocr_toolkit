# from recognition import minimal_text_recognition
try:
    from . import handwritten_text_recognition
    from . import minimal_text_recognition
    from . import tesseract_text_recognition
except:
    from recognition import handwritten_text_recognition
    from recognition import minimal_text_recognition
    from recognition import tesseract_text_recognition
