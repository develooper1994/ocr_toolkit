try:
    import darknet
except:
    try:
        from Darknet import darknet
    except:
        try:
            from LPR.Darknet import darknet
        except:
            from detection.LPR.Darknet import darknet