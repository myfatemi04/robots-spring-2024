from dataclasses import dataclass
import torch
import numpy as np
from sklearn.svm import SVC
from typing import Tuple, List

# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.predict
class MemoryKey:
    """
    Takes in a set of "training examples", which are just sets of
    positive and negative feature vectors. We'll balance the class
    weights so they're roughly even.
    """
    def __init__(self, positive_examples: list, negative_examples: list):
        self.positive_examples = positive_examples # or []
        self.negative_examples = negative_examples # or []
        self.svm = self._fit_svm()
        
    def _fit_svm(self):
        X = np.stack(self.positive_examples + self.negative_examples, axis=0)
        y = np.array([
            int(i < len(self.positive_examples))
            for i in range(
                len(self.positive_examples) + len(self.negative_examples)
            )
        ])
        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)
        svm = SVC(probability=True, class_weight={
            0: len(self.positive_examples)/len(X),
            1: len(self.negative_examples)/len(X),
        }, kernel='linear')
        svm.fit(X, y)
        return svm
        
    # only returns the positive logprob
    def get_logprobs(self, X):
        return self.svm.predict_log_proba(X)[..., 0]
    
    def get_probs(self, X):
        return self.svm.predict_proba(X)[..., 0]

@dataclass
class Memory:
    key: MemoryKey
    """
    represents the scenarios that should activate this memory.
    """
    
    value: str
    """
    represents linguistic knowledge that should be associated
    with this object.
    """

Retrieval = Tuple[float, Memory]

class MemoryBank:
    def __init__(self):
        self.memories = []
    
    def retrieve(self, object_clip_embedding: np.ndarray, topk=None, threshold=None) -> List[Retrieval]:
        similarity_scores = sorted([
            (memory.key.get_probs(object_clip_embedding[None])[0], i)
            for i, memory in enumerate(self.memories)
        ], reverse=True)
        
        return [
            (score, self.memories[i])
            for (score, i) in similarity_scores
            if (topk is not None and (i + 1) <= topk)
            or (threshold is not None and score >= threshold)
        ]
    
    def store(self, memory: Memory):
        self.memories.append(memory)
