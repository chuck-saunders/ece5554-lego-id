from brick_id.dataset.catalog import Brick
from typing import List

def vote_on(self, blob, guesses: List[Brick]):
    votes = dict() #[Brick, int]; not sure why type hinting is warning on this
    other_votes = dict()
    for guess in guesses:
        current_votes = votes.get(guess, 0)
        current_votes += 1
        votes[guess] = current_votes
    if len(votes) == 1:  # Unanimous vote
        return votes
    fifty_percent_threshold = len(guesses)/2
    for vote, count in votes.items():
        if count > fifty_percent_threshold:
            return vote
    # TODO: What should we do if everyone calls it a different thing?
    print('No class got more than 50% of the vote; setting guess to NOT_IN_CATALOG! Guesses were:')
    for guess in guesses:
        print(guess)
    return Brick.NOT_IN_CATALOG
