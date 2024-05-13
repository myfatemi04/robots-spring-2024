
from dataclasses import dataclass

from event_stream import Event, EventStream
from memory_bank_v2 import MemoryBank


@dataclass
class AgentState:
    event_stream: EventStream
    memory_bank: MemoryBank

    def write_event(self, event: Event):
        self.event_stream.write(event)
