from dataclasses import dataclass
from typing import Optional

# Stores information relevant to the current agent state.
# This should contain almost "excessive" information, to allow
# all parts of the system to utilize the entire context.
@dataclass
class WorkingMemory:
    task: str
    plan_natural_language: Optional[str]
    plan_code: Optional[str]

    def serialize_current_plan(self):
        prompt = f"Task: {self.task}\n\n"
        if self.plan_natural_language is not None:
            prompt += f"""You have written the following plan:
\"""
{self.plan_natural_language}
\"""

We are in the middle of the code's execution. The line we are currently executing is
`scene.choose({object_type}, {purpose})`.\n\n"""
            
        return plan

