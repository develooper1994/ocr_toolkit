try:
    import recognition.recognizer
except:
    try:
        import tesseract_text_recognition.recognition.recognizer
    except:
        import recognition.tesseract_text_recognition.recognition.recognizer