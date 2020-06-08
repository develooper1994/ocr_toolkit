# from recognition import minimal_text_recognition
try:
    import handwritten_text_recognition
    import minimal_text_recognition
    import tesseract_text_recognition
except:
    try:
        from recognition import handwritten_text_recognition
        from recognition import minimal_text_recognition
        from recognition import tesseract_text_recognition
    except:
        import recognition.handwritten_text_recognition
        import recognition.minimal_text_recognition
        import recognition.tesseract_text_recognition
