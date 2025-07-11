import textwrap
from dataclasses import dataclass, field


@dataclass
class Prompt:
    id: str
    system: str = field(default="")
    user: str = field(default="")

    def __post_init__(self):
        self.system = textwrap.dedent(self.system)
        self.user = textwrap.dedent(self.user)