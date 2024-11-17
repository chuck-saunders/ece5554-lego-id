# abc = abstract base class
from abc import ABC, abstractmethod
from brick_id.dataset.catalog import Brick

class Solution(ABC):
    @abstractmethod
    def identify(self, blob) -> Brick:
        pass
    def __init__(self):
        pass
