try:
    import recognition.recognizer
except:
    try:
        import minimal_text_recognition.recognition.recognizer
    except:
        import recognition.minimal_text_recognition.recognition.recognizer
