import matplotlib.pyplot as plt
import PIL.Image
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from detect_objects import detect
from select_object_v2 import draw_set_of_marks
import numpy as np


def select_bounding_box(image):
    plt.rcParams['figure.figsize'] = (20, 10)

    def line_select_callback(eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        # print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        # print(" The button you used were: %s %s" % (eclick.button, erelease.button))

    def toggle_selector(event):
        # print(' Key pressed.')
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            # print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            # print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)


    fig, current_ax = plt.subplots()
    # print("\n      click  -->  release")

    # drawtype is 'box' or 'line' or 'none'
    toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                        useblit=True,
                                        button=[1, 3], # type: ignore
                                        minspanx=5, minspany=5,
                                        spancoords='pixels',
                                        interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.imshow(image)
    plt.show()

    # get value
    x1, y1, x2, y2 = toggle_selector.RS.extents
    return int(x1), int(y1), int(x2), int(y2)

def teach_robot():
    # Here, the human provides some direct annotations for what to do (e.g. circling an object and saying what to do with it)
    # Human gives an instruction.
    # Robot asks "OK, could you walk me through it?" or something similar.
    user = "Michael"
    instructions = "Give me a snack"
    im = PIL.Image.open("sample_images/IMG_8650.jpeg")

    detections = detect(im, "snack bag")
    print(detections)
    drawn = draw_set_of_marks(im, detections)
    plt.imshow(drawn)
    plt.axis('off')
    plt.show()

    box = select_bounding_box(im)
    print(box)

teach_robot()
