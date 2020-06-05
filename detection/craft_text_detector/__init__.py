try:
    import craft_text_detector
except:
    try:
        from craft_text_detector import craft_text_detector
    except:
        from detection.craft_text_detector import craft_text_detector
    # except:
    #     from detection import craft_text_detector
