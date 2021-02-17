
import stillleben as sl
import cv2

import pathlib

MY_PATH = pathlib.Path(__file__).parent.absolute()

_C = sl.extension.load(name='picking_heuristic_C', sources=[
    MY_PATH / 'picking_heuristic.cpp',
    MY_PATH / 'pole_of_inaccessibility.cpp',
    MY_PATH / 'graph' / 'graph.cpp',
    MY_PATH / 'graph' / 'tarjan.cpp'
], extra_include_paths=[
    '/usr/include/eigen3',
    '/usr/include/opencv4',
], extra_ldflags=[
    '-lopencv_core',
    '-lopencv_imgproc',
], verbose=True)

postprocess_segmentation = _C.postprocess_segmentation
visualize_detections = _C.visualize_detections
