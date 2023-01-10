from dataclasses import dataclass
from datetime import date

@dataclass
class args:
    #---# Dataset #---#
    dataset : str

    #---# Model #---#
    model : str