try:
    import craft_text_detector
    import LPR
    import detection_selector
except:
    try:
        from detection import craft_text_detector
        from detection import LPR
        from detection import detection_selector
    except:
        from ocr_toolkit.detection import craft_text_detector
        from ocr_toolkit.detection import LPR
        from ocr_toolkit.detection import detection_selector
# from detection.craft_text_detector.craft_text_detector import craft_detector as craft
