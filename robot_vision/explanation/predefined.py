from robot_vision.explanation import segmenters
from robot_vision.explanation import perturbers
from robot_vision.explanation import explainers
from robot_vision.explanation import visualizers


def predefined_selector(expl_method):
    
    # Inexistent explanation method
    if expl_method not in PREDEFINED_EXPLAINERS.keys():
        raise Exception('Invalid explanation method ('+expl_method+'). Must be one of:'+str(PREDEFINED_EXPLAINERS.keys()))
    
    return PREDEFINED_EXPLAINERS[expl_method]


def predefined_lime(img, classifier_fn, label_to_explain=None, num_samples=2000):

    # Segment image
    segmenter = segmenters.SlicSegmenter(img)
    segments = segmenter.segment(n_segments=50, compactness=20, spacing=[1,1])

    # Perturb image
    perturber = perturbers.MultiplePerturber(img, segments, classifier_fn=classifier_fn, hide_color=0)
    data, labels = perturber.perturb(num_samples=num_samples, progress_bar=True)

    # Explain
    explainer = explainers.LimeExplainer(data, labels)
    scores = explainer.explain(label_to_explain=label_to_explain, kernel='lime')

    # Visualize
    visualizer = visualizers.PosNegVisualizer(img, segments, scores)
    score_map = visualizer.get_score_map()
    rgb_score_map = visualizer.visualize(score_map)
    exp_img = visualizer.visualize_on_image(rgb_score_map)

    return exp_img


def predefined_rise(img, classifier_fn, label_to_explain=None, num_samples=2000):

    # Segment image
    segmenter = segmenters.RiseSegmenter(img)
    segments = segmenter.segment(n_seg=[10, 10])

    # Perturb image
    perturber = perturbers.MultiplePerturber(img, segments, classifier_fn=classifier_fn, hide_color=0)
    data, labels = perturber.perturb(num_samples=num_samples, progress_bar=True)

    # Explain
    explainer = explainers.MeanExplainer(data, labels)
    scores = explainer.explain(label_to_explain=label_to_explain)

    # Visualize
    visualizer = visualizers.HeatmapVisualizer(img, segments, scores)
    score_map = visualizer.get_score_map()
    rgb_score_map = visualizer.visualize(score_map)
    exp_img = visualizer.visualize_on_image(rgb_score_map)

    return exp_img


def predefined_kernel_shap(img, classifier_fn, label_to_explain=None, num_samples=2000):

    # Segment image
    segmenter = segmenters.SlicSegmenter(img)
    segments = segmenter.segment(n_segments=50, compactness=20, spacing=[1,1])

    # Perturb image
    perturber = perturbers.ShapPerturber(img, segments, classifier_fn=classifier_fn, hide_color=0)
    perturber.perturb(num_samples=num_samples, progress_bar=True, algorithm='kernel')

    # Explain
    explainer = perturber
    scores = explainer.explain(label_to_explain=label_to_explain)

    # Visualize
    visualizer = visualizers.PosNegVisualizer(img, segments, scores)
    score_map = visualizer.get_score_map()
    rgb_score_map = visualizer.visualize(score_map)
    exp_img = visualizer.visualize_on_image(rgb_score_map)

    return exp_img


def predefined_shap(img, classifier_fn, label_to_explain=None, num_samples=50):

    # Segment image
    segmenter = segmenters.SlicSegmenter(img)
    segments = segmenter.segment(n_segments=50, compactness=20, spacing=[1,1])

    # Perturb image
    perturber = perturbers.AccumPerturber(img, segments, classifier_fn=classifier_fn, hide_color=0)
    data, labels = perturber.perturb(num_samples=num_samples, progress_bar=True)

    # Explain
    explainer = explainers.ShapExplainer(data, labels)
    scores = explainer.explain(label_to_explain=label_to_explain)

    # Visualize
    visualizer = visualizers.PosNegVisualizer(img, segments, scores)
    score_map = visualizer.get_score_map()
    rgb_score_map = visualizer.visualize(score_map)
    exp_img = visualizer.visualize_on_image(rgb_score_map)

    return exp_img


def predefined_occlusion(img, classifier_fn, label_to_explain=None, kernel_size=[5, 5], stride=[10, 10]):

    # Perturb image
    perturber = perturbers.ConvolutionalPerturber(img, classifier_fn=classifier_fn, hide_color=0, kernel_size=kernel_size, stride=stride)
    data, labels = perturber.perturb(progress_bar=True)

    # Explain
    explainer = explainers.ValueExplainer(data, labels)
    scores = explainer.explain(label_to_explain=label_to_explain)

    # Visualize
    visualizer = visualizers.ConvolutionalVisualizer(img, scores, kernel_size=kernel_size, stride=stride)
    score_map = visualizer.get_score_map()
    rgb_score_map = visualizer.visualize(score_map)
    exp_img = visualizer.visualize_on_image(rgb_score_map)

    return exp_img


def predefined_loco(img, classifier_fn, label_to_explain=None):

    # Segment image
    segmenter = segmenters.SlicSegmenter(img)
    segments = segmenter.segment(n_segments=20, compactness=20, spacing=[1,1])

    # Perturb image
    perturber = perturbers.SinglePerturber(img, segments, classifier_fn=classifier_fn, hide_color=0)
    data, labels = perturber.perturb(progress_bar=True)

    # Explain
    explainer = explainers.DifferenceExplainer(data, labels)
    scores = explainer.explain(label_to_explain=label_to_explain)

    # Visualize
    visualizer = visualizers.PosNegVisualizer(img, segments, scores)
    score_map = visualizer.get_score_map()
    rgb_score_map = visualizer.visualize(score_map)
    exp_img = visualizer.visualize_on_image(rgb_score_map)

    return exp_img


def predefined_univariate(img, classifier_fn, label_to_explain=None):

    # Segment image
    segmenter = segmenters.SlicSegmenter(img)
    segments = segmenter.segment(n_segments=20, compactness=20, spacing=[1,1])

    # Perturb image
    perturber = perturbers.AllButOnePerturber(img, segments, classifier_fn=classifier_fn, hide_color=0)
    data, labels = perturber.perturb(progress_bar=True)

    # Explain
    explainer = explainers.DifferenceExplainer(data, labels)
    scores = explainer.explain(label_to_explain=label_to_explain, invert=True)

    # Visualize
    visualizer = visualizers.PosNegVisualizer(img, segments, scores)
    score_map = visualizer.get_score_map()
    rgb_score_map = visualizer.visualize(score_map)
    exp_img = visualizer.visualize_on_image(rgb_score_map)

    return exp_img


PREDEFINED_EXPLAINERS = {
    'lime': predefined_lime,
    'rise': predefined_rise,
    'kernel_shap': predefined_kernel_shap,
    'shap': predefined_shap,
    'occlusion_sensitivity': predefined_occlusion,
    'univariate': predefined_univariate,
    'loco': predefined_loco,
}