try:
    from recognition import recognizer
except:
    try:
        from tesseract_text_recognition.recognition import recognizer
    except:
        from recognition.tesseract_text_recognition.recognition import recognizer
