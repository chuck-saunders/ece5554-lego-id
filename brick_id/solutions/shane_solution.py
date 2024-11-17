from .solution import Solution
from brick_id.dataset.catalog import Brick

class ShaneSolution(Solution):
    def identify(self, blob):
        # TODO: Write your implementation
        return Brick.NOT_IN_CATALOG
