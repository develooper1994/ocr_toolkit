try:
    import data_generator
    import drawing_utils
    import keras_utils
    import label
    import loss
    import projection_utils
    import sampler
    import utils
except:
    try:
        from src import data_generator
        from src import drawing_utils
        from src import keras_utils
        from src import label
        from src import loss
        from src import projection_utils
        from src import sampler
        from src import utils
    except:
        try:
            from platerecognition.src import data_generator
            from platerecognition.src import drawing_utils
            from platerecognition.src import keras_utils
            from platerecognition.src import label
            from platerecognition.src import loss
            from platerecognition.src import projection_utils
            from platerecognition.src import sampler
            from platerecognition.src import utils
        except:
            try:
                from LPR.platerecognition.src import data_generator
                from LPR.platerecognition.src import drawing_utils
                from LPR.platerecognition.src import keras_utils
                from LPR.platerecognition.src import label
                from LPR.platerecognition.src import loss
                from LPR.platerecognition.src import projection_utils
                from LPR.platerecognition.src import sampler
                from LPR.platerecognition.src import utils
            except:
                from detection.LPR.platerecognition.src import data_generator
                from detection.LPR.platerecognition.src import drawing_utils
                from detection.LPR.platerecognition.src import keras_utils
                from detection.LPR.platerecognition.src import label
                from detection.LPR.platerecognition.src import loss
                from detection.LPR.platerecognition.src import projection_utils
                from detection.LPR.platerecognition.src import sampler
                from detection.LPR.platerecognition.src import utils