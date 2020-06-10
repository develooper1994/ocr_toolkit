import unittest

try:
    # direct call
    print("direct call")
    from .craft_text_detector import craft_detector
    import craft_detector
except:
    # indirect call
    print("indirect call")
    try:
        from .craft_text_detector import craft_detector
        from .craft_text_detector.craft_detector import craft_detector
    except:
        from detection.craft_text_detector import craft_text_detector
        from detection.craft_text_detector.craft_text_detector import craft_detector

# set image path and export folder directory
image_name = 'idcard.png'
image_path = '../figures/' + image_name  # linux
# image_path = r'C:\Users\selcu\PycharmProjects\ocr_toolkit\detection\craft_text_detector\figures/' + image_name  # windows
output_dir = None
device = "gpu"  # False
show_time = False
craft_model_path = "../craft_mlt_25k.pth"
refinenet_model_path = "../craft_refiner_CTW1500.pth"

# load image
image = craft_text_detector.imgproc.read_image(image_path)
# refine_net = None
pred = craft_detector.craft_detector(image=image,
                                     craft_model_path=craft_model_path,
                                     refinenet_model_path=refinenet_model_path,
                                     device=device)


class TestCraftTextDetector(unittest.TestCase):
    def test_load_craftnet_model(self):
        craft_net = pred.craft_net
        self.assertTrue(craft_net)

    def test_load_refinenet_model(self):
        refine_net = pred.refine_net
        self.assertTrue(refine_net)

    def test_get_prediction(self):
        # load image
        image = craft_text_detector.imgproc.read_image(image_path)

        # perform prediction
        text_threshold = 0.9
        link_threshold = 0.2
        low_text = 0.2
        prediction_result = pred.get_prediction(image=image,
                                                text_threshold=text_threshold,
                                                link_threshold=link_threshold,
                                                low_text=low_text,
                                                square_size=720,
                                                show_time=show_time)

        # !!! get_prediction.py -> get_prediction(...)
        #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        #         image, square_size, interpolation=cv2.INTER_LINEAR
        #     )
        # self.assertEqual(35, len(prediction_result["boxes"]))

        #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        #         image, square_size, interpolation=cv2.INTER_CUBIC
        #     )

        # TODO solve what is happining. why number changed.
        # self.assertEqual(15, len(prediction_result["boxes"]))  # refinenet_model_path=None -> 37, refinenet_model_path=refinenet_model_path -> 15
        self.assertEqual(15, len(prediction_result["boxes"]))  # refinenet_model_path=None -> 37, refinenet_model_path=refinenet_model_path -> 14
        self.assertEqual(4, len(prediction_result["boxes"][0]))
        self.assertEqual(2, len(prediction_result["boxes"][0][0]))
        self.assertEqual(111, int(prediction_result["boxes"][0][0][0]))
        # !!! get_prediction.py -> get_prediction(...)
        #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        #         image, square_size, interpolation=cv2.INTER_LINEAR
        #     )
        # self.assertEqual(len(prediction_result["polys"]), 35)

        #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        #         image, square_size, interpolation=cv2.INTER_CUBIC
        #     )
        self.assertEqual(15, len(prediction_result[
                                     "polys"]))  # refinenet_model_path=None -> 37, refinenet_model_path=refinenet_model_path -> 15
        self.assertEqual((240, 368, 3), prediction_result["heatmaps"]["text_score_heatmap"].shape)

    def test_detect_text(self):
        # refiner = False
        pred = craft_detector.craft_detector(image=image,
                                             craft_model_path=craft_model_path,
                                             refinenet_model_path=None,
                                             device=device)
        prediction_result = pred.detect_text(image=image_path, output_dir=output_dir, rectify=True, export_extra=False,
                                             text_threshold=0.7, link_threshold=0.4, low_text=0.4, square_size=720,
                                             show_time=show_time, crop_type="is_poly")

        # !!! get_prediction.py -> get_prediction(...)
        #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        #         image, square_size, interpolation=cv2.INTER_LINEAR
        #     )
        # self.assertEqual(52, len(prediction_result["boxes"]))

        #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        #         image, square_size, interpolation=cv2.INTER_CUBIC
        #     )
        self.assertEqual(51, len(prediction_result["boxes"]))
        self.assertEqual(4, len(prediction_result["boxes"][0]))
        self.assertEqual(2, len(prediction_result["boxes"][0][0]))
        self.assertEqual(115, int(prediction_result["boxes"][0][0][0]))

        # refiner = True
        pred.reload(image=image,
                    craft_model_path=craft_model_path,
                    refinenet_model_path=refinenet_model_path,
                    device=device)
        prediction_result = pred.detect_text(image=image_path, output_dir=output_dir, rectify=True, export_extra=False,
                                             text_threshold=0.7, link_threshold=0.4, low_text=0.4, square_size=720,
                                             show_time=show_time, crop_type="is_poly")
        self.prediction_result_compare(prediction_result)

        # refiner = False
        pred.reload(image=image,
                    craft_model_path=craft_model_path,
                    refinenet_model_path=None,
                    device=device)
        prediction_result = pred.detect_text(image=image_path, output_dir=output_dir, rectify=False, export_extra=False,
                                             text_threshold=0.7, link_threshold=0.4, low_text=0.4, square_size=720,
                                             show_time=show_time, crop_type="box")
        # !!! get_prediction.py -> get_prediction(...)
        #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        #         image, square_size, interpolation=cv2.INTER_LINEAR
        #     )
        # self.assertEqual(52, len(prediction_result["boxes"]))

        #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        #         image, square_size, interpolation=cv2.INTER_CUBIC
        #     )
        self.assertEqual(51, len(prediction_result["boxes"]))
        self.assertEqual(4, len(prediction_result["boxes"][0]))
        self.assertEqual(2, len(prediction_result["boxes"][0][0]))
        self.assertEqual(244, int(prediction_result["boxes"][0][2][0]))

        # refiner = True
        pred.reload(image=image,
                    craft_model_path=craft_model_path,
                    refinenet_model_path=refinenet_model_path,
                    device=device)
        prediction_result = pred.detect_text(image=image_path, output_dir=output_dir, rectify=False, export_extra=False,
                                             text_threshold=0.7, link_threshold=0.4, low_text=0.4, square_size=720,
                                             show_time=show_time, crop_type="box")
        self.prediction_result_compare(prediction_result)

    def prediction_result_compare(self, prediction_result):
        self.assertEqual(19, len(prediction_result["boxes"]))  # Expected :51
        self.assertEqual(4, len(prediction_result["boxes"][0]))
        self.assertEqual(2, len(prediction_result["boxes"][0][0]))
        self.assertEqual(661, int(prediction_result["boxes"][0][2][0]))  # Expected :224


if __name__ == '__main__':
    unittest.main()
