try:
    import recognition.recognizer
except:
    try:
        from minimal_text_recognition.recognition import recognizer
    except:
        from recognition.minimal_text_recognition.recognition import recognizer
