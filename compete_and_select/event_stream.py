from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

import numpy as np
import PIL.Image
from detect_objects import Detection


class EventType(Enum):
    VISUAL_PERCEPTION = 0
    INNER_MONOLOGUE = 1
    CODE_ACTION = 2
    VERBAL_FEEDBACK = 3

@dataclass
class Event:
    pass
    # type: EventType

@dataclass
class EventStream:
    events: List[Event] = field(default_factory=list)

    def write(self, event: Event):
        self.events.append(event)

@dataclass
class VisualPerceptionEvent(Event):
    imgs: List[PIL.Image.Image]
    pcds: List[Optional[np.ndarray]]

@dataclass
class CodeActionEvent(Event):
    rationale: str
    code: Optional[str]
    raw_content: str
    # Should store the values of anything that changes (e.g. freeze the state of the environment)

@dataclass
class ReflectionEvent(Event):
    # We reflect on an action.
    # For example, maybe we generate memories associated with certain objects.
    # Then we log the human feedback associated with those memories.
    # How do we refer to the objects though?
    reflection: str

@dataclass
class VerbalFeedbackEvent(Event):
    text: str
    variables: Optional[dict] = None

@dataclass
class ExceptionEvent(Event):
    exception_type: str
    text: str


@dataclass
class FunctionCallEnter(Event):
    function_name: str
    args: list
    kwargs: dict = field(default_factory=dict)

@dataclass
class FunctionCallResult(Event):
    result: Any

@dataclass
class VariableUpdate(Event):
    variable_name: str
    new_value: Any

@dataclass
class ObjectSelectionInitiation(Event):
    object_type: str
    object_purpose: str

@dataclass
class ObjectSelectionDetectionResult(Event):
    detections: List[Detection]

@dataclass
class ObjectSelectionPolicyCreation(Event):
    rationale: str
    logits: np.ndarray

@dataclass
class ObjectSelectionPolicySelection(Event):
    selected_object_id: int
