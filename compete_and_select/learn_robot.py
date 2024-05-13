import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageFilter
from detect_objects import detect, embed_box, get_clip_embedding_map
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from memory_bank_v2 import MemoryBank
from select_object_v2 import draw_set_of_marks
from sklearn.svm import SVC


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
    x1, x2, y1, y2 = toggle_selector.RS.extents
    return int(x1), int(y1), int(x2), int(y2)

def teach_robot():
    memory_bank = MemoryBank()

    # Here, the human provides some direct annotations for what to do (e.g. circling an object and saying what to do with it)
    # Human gives an instruction.
    # Robot asks "OK, could you walk me through it?" or something similar.
    user = "Michael"
    instructions = "Put the snacks in the bowl"
    image = PIL.Image.open("sample_images/IMG_8650.jpeg")

    # detections = detect(image, "snack bag")
    # print(detections)
    # drawn = draw_set_of_marks(image, detections)
    # plt.imshow(drawn)
    # plt.axis('off')
    # plt.show()

    embed_grid = get_clip_embedding_map(image).detach().cpu().numpy()

    image_blur = image.filter(PIL.ImageFilter.GaussianBlur(2))
    embed_grid_2 = get_clip_embedding_map(image_blur).detach().cpu().numpy()

    box_embed = embed_grid[8, 5]

    box = select_bounding_box(image)
    embeds_inside_box = []
    embeds_outside_box = []

    for embed_grid in [embed_grid, embed_grid_2]:
        for row in range(16):
            for col in range(16):
                x = col * 14 * (image.width / 224)
                y = row * 14 * (image.height / 224)
                if (box[0] <= x <= box[2]) and (box[1] <= y <= box[3]):
                    embeds_inside_box.append(embed_grid[row, col])
                else:
                    embeds_outside_box.append(embed_grid[row, col])

    embeds = np.stack(embeds_inside_box + embeds_outside_box)
    class_labels = [1]*len(embeds_inside_box) + [0]*len(embeds_outside_box)
    # sample_weights = [1]*len(embeds_inside_box) + [len(embeds_inside_box)/len(embeds_outside_box)]*len(embeds_outside_box)
    svm = SVC(probability=True)
    svm = svm.fit(embeds, class_labels) # , sample_weights)

    # NEW IMAGE TIME!!!
    image = PIL.Image.open("sample_images/IMG_8651.jpeg")
    embed_grid = get_clip_embedding_map(image).detach().cpu().numpy()

    grid_match_score = svm.predict_proba(embed_grid.reshape(-1, 1024))[:, 1].reshape(16, 16)

    print(grid_match_score)

    # grid_match_score = (embed_grid * box_embed).sum(axis=-1)/(np.linalg.norm(embed_grid, axis=-1)*np.linalg.norm(box_embed))
    # grid_match_score = grid_match_score - grid_match_score.min()
    # grid_match_score = grid_match_score / grid_match_score.max()

    match_image = PIL.Image.fromarray(np.uint8(grid_match_score * 255)).resize((image.width, image.height))
    match_image = np.array(match_image) / 255

    plt.imshow(image)
    plt.imshow(match_image, alpha=match_image)

    # plt.imshow(match_image)
    plt.axis('off')
    plt.show()


teach_robot()
