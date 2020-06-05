try:
    import Darknet
    import platerecognition
except:
    try:
        from LPR import Darknet
        from LPR import platerecognition
    except:
        from detection.LPR import Darknet
        from detection.LPR import platerecognition