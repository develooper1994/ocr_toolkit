try:
    import recognition
except:
    try:
        from handwritten_text_recognition import recognition
    except:
        from recognition.handwritten_text_recognition import recognition
