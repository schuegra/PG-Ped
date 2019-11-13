from typing import List, NamedTuple
import gc

import random


def sample_random(memory, batch_size: int, position: int) -> List[NamedTuple]:
    number_samples = min(batch_size, len(memory))
    return random.sample(memory, number_samples)


def sample_episode(memory: List, time_steps: int, episode_length: int, position: int) -> List[NamedTuple]:
    underreach = time_steps - position
    if underreach > 0:
        episode_samples = memory[-underreach:]
        episode_samples += memory[:position]
    else:
        episode_samples = memory[position - time_steps:position]

    # Free memory
    for x in memory:
        del x # dereference
    gc.collect() # throw dereferenced objects "out of the window"

    position = 0
    return episode_samples


def sample_augmented(memory, time_steps: int, batch_size: int, position: int) -> List[NamedTuple]:
    '''
        Some of the samples are random, the others are the previous time interval
    '''
    number_samples = min(batch_size, len(memory))
    random_samples = random.sample(memory, number_samples)
    #recent_samples = memory[-time_steps:]
    return random_samples# + recent_samples