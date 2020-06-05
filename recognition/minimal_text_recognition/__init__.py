try:
    import recognition
    import create_lmdb_dataset
    import dataset
    import demo
    import model
    import modules
    import test
    import train
    import utils
except:
    try:
        from minimal_text_recognition import recognition
        from minimal_text_recognition import create_lmdb_dataset
        from minimal_text_recognition import dataset
        from minimal_text_recognition import demo
        from minimal_text_recognition import model
        from minimal_text_recognition import modules
        from minimal_text_recognition import test
        from minimal_text_recognition import train
        from minimal_text_recognition import utils
    except:
        from recognition.minimal_text_recognition import recognition
        from recognition.minimal_text_recognition import create_lmdb_dataset
        from recognition.minimal_text_recognition import dataset
        from recognition.minimal_text_recognition import demo
        from recognition.minimal_text_recognition import model
        from recognition.minimal_text_recognition import modules
        from recognition.minimal_text_recognition import test
        from recognition.minimal_text_recognition import train
        from recognition.minimal_text_recognition import utils

# from recognition.minimal_text_recognition import create_lmdb_dataset
# from recognition.minimal_text_recognition import dataset
# from recognition.minimal_text_recognition import demo
# from recognition.minimal_text_recognition import model
# from recognition.minimal_text_recognition import test
# from recognition.minimal_text_recognition import train
# from recognition.minimal_text_recognition import utils
# from recognition.minimal_text_recognition import modules
