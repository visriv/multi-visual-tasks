import cv2
from enum import Enum
import numpy as np
import os


class Color(Enum):
    """An enum that defines common colors.
    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def color_val(color):
    """Convert various input to color tuples.
    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs
    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if isinstance(color, str):
        return Color[color].value
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    elif isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    else:
        raise TypeError(f'Invalid type for color: {type(color)}')


def imshow_bboxes(img_path,
                  bboxes,
                  color_name='green',
                  top_k=-1,
                  thickness=1,
                  win_name='img_box_show',
                  wait_time=0):
    """Draw bboxes on an image."""
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    
    color = color_val(color_name)
    
    #for i, _bboxes in enumerate(bboxes):
    _bboxes = np.array(bboxes, dtype=np.int32)
    if len(_bboxes) > 0:
        _bboxes[:, 0] = _bboxes[:, 0] * w / 416
        _bboxes[:, 1] = _bboxes[:, 1] * h / 416
        _bboxes[:, 2] = _bboxes[:, 2] * w / 416
        _bboxes[:, 3] = _bboxes[:, 3] * h / 416
    if top_k <= 0:
        _top_k = _bboxes.shape[0]
    else:
        _top_k = min(top_k, _bboxes.shape[0])
    for j in range(_top_k):
        left_top = (_bboxes[j, 0], _bboxes[j, 1])
        right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
        cv2.rectangle(
            img, left_top, right_bottom, color, thickness=thickness)

    cv2.imshow(win_name, img)
    cv2.waitKey(wait_time)
    
    # return img


if __name__ == '__main__':

    test_img_name_list = ['a0524powtdi_019.jpg', '0.jpg', '1.jpg', '2.png', '3.jpg', '4.jpg']
    test_img_dir = 'meta/test_data'

    animal_results = [[[ 31.377762, 3.2548523, 388.96722, 408.88193]],
                      [],
                      [[133.36209, 108.505844, 282.10587, 391.042]],
                      [[ 69.87598, 12.063904, 294.53406, 405.68445]],
                      [[259.17798, 118.24445, 390.48553, 252.91727]],
                      []]
    
    catdog_results = [[[ 51.346947, 8.16761, 297.40515, 214.93607]],
                      [],
                      [[155.90741, 113.888115, 259.36703, 243.53397]],
                      [[212.61664, 27.2159, 285.0178, 147.10396]],
                      [[309.18484, 112.48157, 373.96628, 188.33856]],
                      []]
    multiobj_results = [[[45.487335, 13.523758, 313.34225, 403.15112]],
                        [[20.526062, 71.6855, 129.64767, 421.32654],
                         [88.08489, 90.180786, 187.50928, 412.24814],
                         [154.99467, 87.2697, 301.21326, 415.56012],
                         [303.7769, 90.21428, 408.2361, 409.16968]], 
                        [[133.60712, 106.44365, 283.60837, 387.41925]],
                        [],
                        [[ 96.94179, 105.38433, 190.65361, 322.17786]],
                        [[145.32074, 146.6388, 217.92847, 228.6564]]]

                      
    animal_label = [['french-bulldog'], 
                    [],
                    ['border-collie'],
                    ['cheetah'],
                    ['lion'],
                    []]

    catdog_label = [['dog'], 
                    [],
                    ['dog'],
                    ['cat'],
                    ['dog'],
                    []]
    
    multiobj_label = [['dog'], 
                      ['person', 'person', 'person', 'person'],
                      ['dog'],
                      [],
                      ['person'],
                      ['cartoon-person']]

    for i in range(len(test_img_name_list)):

        img_path = os.path.join(test_img_dir, test_img_name_list[i])
        imshow_bboxes(img_path, animal_results[i])
        imshow_bboxes(img_path, catdog_results[i])
        imshow_bboxes(img_path, multiobj_results[i])
