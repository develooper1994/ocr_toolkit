try:
    import feature_extraction
    import prediction
    import sequence_modeling
    import transformation
except:
    try:
        from modules import feature_extraction
        from modules import prediction
        from modules import sequence_modeling
        from modules import transformation
    except:
        try:
            from minimal_text_recognition.modules import feature_extraction
            from minimal_text_recognition.modules import prediction
            from minimal_text_recognition.modules import sequence_modeling
            from minimal_text_recognition.modules import transformation
        except:
            from recognition.minimal_text_recognition.modules import feature_extraction
            from recognition.minimal_text_recognition.modules import prediction
            from recognition.minimal_text_recognition.modules import sequence_modeling
            from recognition.minimal_text_recognition.modules import transformation

# from recognition.minimal_text_recognition.modules import feature_extraction
# from recognition.minimal_text_recognition.modules import prediction
# from recognition.minimal_text_recognition.modules import sequence_modeling
# from recognition.minimal_text_recognition.modules import transformation
