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
        from ocr_toolkit.recognition import handwritten_text_recognition
        from ocr_toolkit.recognition import minimal_text_recognition
        from ocr_toolkit.recognition import tesseract_text_recognition
