try:
    import rrc_evaluation_funcs
    import script
except:
    try:
        from eval import rrc_evaluation_funcs
        from eval import script
    except:
        try:
            from craft_text_detector.eval import rrc_evaluation_funcs
            from craft_text_detector.eval import script
        except:
            try:
                from craft_text_detector.craft_text_detector.eval import rrc_evaluation_funcs
                from craft_text_detector.craft_text_detector.eval import script
            except:
                from detection.craft_text_detector.craft_text_detector.eval import rrc_evaluation_funcs
                from detection.craft_text_detector.craft_text_detector.eval import script