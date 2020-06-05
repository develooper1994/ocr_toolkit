try:
    import basenet
    import craftnet
    import refinenet
except:
    try:
        from models import basenet
        from models import craftnet
        from models import refinenet
    except:
        try:
            from craft_text_detector.models import basenet
            from craft_text_detector.models import craftnet
            from craft_text_detector.models import refinenet
        except:
            try:
                from craft_text_detector.craft_text_detector.models import basenet
                from craft_text_detector.craft_text_detector.models import craftnet
                from craft_text_detector.craft_text_detector.models import refinenet
            except:
                from detection.craft_text_detector.craft_text_detector.models import basenet
                from detection.craft_text_detector.craft_text_detector.models import craftnet
                from detection.craft_text_detector.craft_text_detector.models import refinenet