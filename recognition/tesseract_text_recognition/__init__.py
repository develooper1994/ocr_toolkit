try:
    import recognition
except:
    try:
        from tesseract_text_recognition import recognition
    except:
        from recognition.tesseract_text_recognition import recognition
