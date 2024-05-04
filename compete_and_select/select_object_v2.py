"""
aims to be an improved version upon set of mark prompting, to address the failure mode of incorrect object choice
"""

def describe_objects_with_retrievals(detections, object_memory_bank):
    # assume that the detections have already had clip embeddings added to them
    nretrievals = []
    prompt_string = 'Detections\n'
    for i, detection in enumerate(detections):
        box = detection['box']
        center_x = (box['xmin'] + box['xmax']) / 2
        center_y = (box['ymin'] + box['ymax']) / 2
        
        prompt_string += f"({i + 1}) at ({center_x:.0f}, {center_y:.0f})\n"
        
        retrievals = object_memory_bank.retrieve(detection['emb'].detach().cpu().numpy(), threshold=0.8)
        if len(retrievals) > 0:
            for score, memory in retrievals:
                addl = f" - Note: This object has a visual similarity score of {score:.2f} to something which you noted, \"{memory.value}\".\n"
                prompt_string += addl
        nretrievals.append(len(retrievals))
                
    return prompt_string, nretrievals
