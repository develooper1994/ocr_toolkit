try:
    import recognition
except:
    try:
        import tesseract_text_recognition.recognition
    except:
        import recognition.tesseract_text_recognition.recognition
