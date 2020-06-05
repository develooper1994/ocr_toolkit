# TODO! implement training loop
try:
    import coordinates
    import data_loader
    import evaluation
    import gaussian
    import mep
    import mseloss
    import torchutil
    import trainSyndata
    import train_MLT_data
    import trainic15data
    import watershed
except:
    try:
        from craft_text_detector.craft_text_detector.train import coordinates
        from craft_text_detector.craft_text_detector.train import data_loader
        from craft_text_detector.craft_text_detector.train import evaluation
        from craft_text_detector.craft_text_detector.train import gaussian
        from craft_text_detector.craft_text_detector.train import mep
        from craft_text_detector.craft_text_detector.train import mseloss
        from craft_text_detector.craft_text_detector.train import torchutil
        from craft_text_detector.craft_text_detector.train import trainSyndata
        from craft_text_detector.craft_text_detector.train import train_MLT_data
        from craft_text_detector.craft_text_detector.train import trainic15data
        from craft_text_detector.craft_text_detector.train import watershed
    except:
        from detection.craft_text_detector.craft_text_detector.train import coordinates
        from detection.craft_text_detector.craft_text_detector.train import data_loader
        from detection.craft_text_detector.craft_text_detector.train import evaluation
        from detection.craft_text_detector.craft_text_detector.train import gaussian
        from detection.craft_text_detector.craft_text_detector.train import mep
        from detection.craft_text_detector.craft_text_detector.train import mseloss
        from detection.craft_text_detector.craft_text_detector.train import torchutil
        from detection.craft_text_detector.craft_text_detector.train import trainSyndata
        from detection.craft_text_detector.craft_text_detector.train import train_MLT_data
        from detection.craft_text_detector.craft_text_detector.train import trainic15data
        from detection.craft_text_detector.craft_text_detector.train import watershed
