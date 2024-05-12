import PIL.Image
import flask

def teach_robot():
    # Here, the human provides some direct annotations for what to do (e.g. circling an object and saying what to do with it)
    # Human gives an instruction.
    # Robot asks "OK, could you walk me through it?" or something similar.
    user = "Michael"
    instructions = "Give me a snack"
    im = PIL.Image.open("IMG_8650.jpeg")

    # generate CLIP embeddings for each part of the image

