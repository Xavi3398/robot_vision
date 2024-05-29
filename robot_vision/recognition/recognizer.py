
class Recognizer:

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_result(img):
        pass

    @staticmethod
    def get_plot_result(img, result):
        pass

    def get_explanation(img, explainer, label=None):
        raise NotImplementedError('This method does not support explanations yet.')