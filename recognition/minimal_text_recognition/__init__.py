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
        import minimal_text_recognition.recognition
        import minimal_text_recognition.create_lmdb_dataset
        import minimal_text_recognition.dataset
        import minimal_text_recognition.demo
        import minimal_text_recognition.model
        import minimal_text_recognition.modules
        import minimal_text_recognition.test
        import minimal_text_recognition.train
        import minimal_text_recognition.utils
    except:
        import recognition.minimal_text_recognition.recognition
        import recognition.minimal_text_recognition.create_lmdb_dataset
        import recognition.minimal_text_recognition.dataset
        import recognition.minimal_text_recognition.demo
        import recognition.minimal_text_recognition.model
        import recognition.minimal_text_recognition.modules
        import recognition.minimal_text_recognition.test
        import recognition.minimal_text_recognition.train
        import recognition.minimal_text_recognition.utils

# from recognition.minimal_text_recognition import create_lmdb_dataset
# from recognition.minimal_text_recognition import dataset
# from recognition.minimal_text_recognition import demo
# from recognition.minimal_text_recognition import model
# from recognition.minimal_text_recognition import test
# from recognition.minimal_text_recognition import train
# from recognition.minimal_text_recognition import utils
# from recognition.minimal_text_recognition import modules
