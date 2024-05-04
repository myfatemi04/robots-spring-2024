from dataclasses import dataclass
import torch

@dataclass
class Memory:
    object_clip_embedding: torch.Tensor
    salient_information: str

class PhrasedMemory:
    def __init__(self, format_string: str, format_string_variables: dict):
        self._format_string = format_string
        self._format_string_variables = format_string_variables
    
    def __getattr__(self, attr):
        if attr.startswith("_"):
            return getattr(super(PhrasedMemory, self), attr)
        else:
            return self._format_string_variables[attr]

class MemoryBank:
    def __init__(self):
        self.memories = []
    
    def retrieve(self, object_clip_embedding, topk=None, threshold=None):
        # ideally at some point we have an object-specific trainable threshold (that is enforced to be symmetric, etc.)
        similarity_scores = []
        emb_norm = torch.norm(object_clip_embedding)
        for i, memory in enumerate(self.memories):
            # use cosine similarity (from clip)
            similarity_score = \
                (memory.object_clip_embedding @ object_clip_embedding).item() / \
                (torch.norm(memory.object_clip_embedding)*emb_norm)
            similarity_scores.append(
                (similarity_score, i)
            )
        
        retrievals = set()
        
        # accept anything with a certain threshold
        if threshold is not None:
            for score, i in similarity_scores:
                if score > threshold:
                    retrievals.add(i)
        
        # accept anything in the top k
        if topk is not None and len(retrievals) < topk:
            similarity_scores = sorted(similarity_scores, reverse=True)
            retrievals.update({i for score, i in similarity_scores[:topk]})
            
        return [
            (similarity_scores[i][0], self.memories[i]) for i in retrievals
        ]
    
    def add_object_memory(self, memory: Memory):
        self.memories.append(memory)
        

