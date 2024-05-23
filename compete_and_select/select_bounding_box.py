import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


def select_bounding_box(image, prompt="Select a bounding box"):
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
    plt.title(prompt)
    plt.connect('key_press_event', toggle_selector)
    plt.imshow(image)
    plt.show()

    # get value
    x1, x2, y1, y2 = toggle_selector.RS.extents
    return int(x1), int(y1), int(x2), int(y2)
