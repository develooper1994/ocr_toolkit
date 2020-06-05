try:
    import test_craft
except:
    try:
        from tests import test_craft
    except:
        from craft_text_detector.tests import test_craft
# from detection.craft_text_detector.tests import test_craft