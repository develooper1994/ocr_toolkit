try:
    import recognition
except:
    try:
        import handwritten_text_recognition.recognition
    except:
        import recognition.handwritten_text_recognition.recognition
