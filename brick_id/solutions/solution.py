# abc = abstract base class
from abc import ABC, abstractmethod


class Solution(ABC):
    @abstractmethod
    def get_objects(self):
        pass
    def __init__(self):
        pass
