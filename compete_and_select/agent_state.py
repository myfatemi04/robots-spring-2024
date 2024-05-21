
from dataclasses import dataclass
from typing import Optional

from event_stream import Event, EventStream
from memory_bank_v2 import MemoryBank
from rgbd_asynchronous_tracker import RGBDAsynchronousTracker


@dataclass
class AgentState:
    event_stream: EventStream
    memory_bank: MemoryBank
    tracker: Optional[RGBDAsynchronousTracker]

    def write_event(self, event: Event):
        self.event_stream.write(event)

    @property
    def most_recent_visual_observation(self):
        from event_stream import VisualPerceptionEvent

        i = len(self.event_stream.events) - 1
        while i >= 0:
            event = self.event_stream.events[i]
            if isinstance(event, VisualPerceptionEvent):
                return event
            
            i -= 1

        return None
