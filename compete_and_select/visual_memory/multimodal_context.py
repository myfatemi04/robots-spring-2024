"""
Let's say we want to have a long multimodal context, centered around specific objects.
In such a case, it might be hard to perform retrieval on this context, and may be unnecessarily
expensive (data-hungry or otherwise) to train a model to perform retrieval on this context.

I have a suspicion that most technologies to generate long-context training data would be synthetic.
Additionally, such models *don't need knowledge* in order to learn. They do not need data beyond
basic English reasoning, which is pretrained. Thus; I have a suspicion that it would be a waste
of resources to train a large transformer for something that is effectively meta-learning.

The training data as input would be a set of objects to be used as training examples, and the
corresponding knowledge.

Additionally, we might be able to do "speculative decoding" for robotics. In which case we can
simply ask a vision-language model to describe the relevant object state, and verify whether
a task's execution has been interrupted. If it hasn't been interrupted, then the next step of
the code can be executed.

Time to do some prompt engineering... If there are a ton of memories associated with particular
objects, then maybe we can rank them.

ALSO... What about just having general domain knowledge associated with certain objects? E.g.
if we take a ton of training examples for a 'phone' from some traditional dataset... can we
generate a visual key-value knowledge base?

What if there are facts that overlap between objects?

Also, how much does accuracy improve for using this sort of object-centric retrieval compared
to direct prompting? Could we chain this improved type of object recognition with tool use to
generate a bunch of knowledge?

It needs to be really really easy to collect additional training examples.

Ah and so I think I can finally understand "the point". I am actually creating a few-shot
object recognition platform.

Few-shot object recognition in 2D feature fields?

There is no benefit to few-shot object recognition, as almost everything is already included
in most generalist models. ... or is there.

Paper title: "Object memory as few-shot object recognition"
Dummy task: "Which of these is mine?" or "Which if these could be useful for [x]?"
Retrieval augmentation is still a nice grounding technique... For example I could
include a large dataset of objects and useful information for how to manipulate them.

"""

# Let's just try to construct a simple retrieval mechanism.
# PaliGemma is the optimal method imo though. Could select objects from images.

import PIL.Image
from .. import dense_clip_embeddings

image = PIL.Image.open("sample_images/chargers.png")

ctx = [
    image,
    "Which of these objects would be useful for charging a phone?",
]

embeds = dense_clip_embeddings.get_clip_embeddings_dense(image)

