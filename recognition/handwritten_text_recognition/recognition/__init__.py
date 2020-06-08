try:
    import recognition.recognizer
    import recognition.get_models
    import recognition.ocr
    import recognition.tests
    import recognition.utils
except:
    try:
        from handwritten_text_recognition.recognition import recognizer
        from handwritten_text_recognition.recognition import get_models
        from handwritten_text_recognition.recognition import ocr
        from handwritten_text_recognition.recognition import tests
        from handwritten_text_recognition.recognition import utils
    except:
        from recognition.handwritten_text_recognition.recognition import recognizer
        from recognition.handwritten_text_recognition.recognition import get_models
        from recognition.handwritten_text_recognition.recognition import ocr
        from recognition.handwritten_text_recognition.recognition import tests
        from recognition.handwritten_text_recognition.recognition import utils