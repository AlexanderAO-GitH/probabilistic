# %%
from collections import Counter

import pyro
import torch
from tqdm import tqdm
import numpy as np


# %%
def coin_toss(n):
    faces = Counter()
    total = 0
    for _ in tqdm(range(n)):

        # create a sample of Bernoulli distribution for fair (50/50) coin
        # The size of the sample is 1
        toss_result = np.random.choice(["head", "tail"])

        # convert the Bernoulli distribution into meaning
        # what does 1/0 stands for ?
        # it meeans it can only show one face after toss
        # gain 2 if head is tossed, otherwise lose 2
        reward = {"head": 2, "tail": -2}
        total += reward[toss_result]

        # update the faces Counter
        faces[toss_result] += 1
    return faces, total


def coin_toss_tensor(n):

    # create a sample of Bernoulli distribution for fair (50/50) coin
    # The size of the sample is n
    toss_results = torch.randint(0, 2, size=(n,), dtype=torch.int32)
    toss_results_list = toss_results.tolist()

    # return Counter object to summarize the counts of head/tail, and total rewards
    faces = Counter(toss_results_list)
    
    # Calculate total rewards
    total_reward = (toss_results * 4 - 2 * n).sum().item()
    return faces, total_reward


def simulation(n, simulation_func):
    faces, total = simulation_func(n)

    print(f"\nRan {n} simulation{'s' if n >1 else ''}")
    print(f"Total Reward = {total}")
    print(faces)


# %%
from collections import Counter
from tqdm import tqdm
import numpy as np
def coin_toss(n):
    faces = Counter()
    total = 0
    for _ in tqdm(range(n)):
        toss_result = np.random.choice(["head", "tail"])

        reward = {"head": 2, "tail": -2}
        total += reward[toss_result]

        faces[toss_result] += 1
    return faces, total
def simulation(n, simulation_func):
    faces, total = simulation_func(n)

    print(f"\nRan {n} simulation{'s' if n >1 else ''}")
    print(f"Total Reward = {total}")
    print(faces)

for n in [1, 1000, 1000000]:
    simulation(n, coin_toss)


# %%
from collections import Counter
from tqdm import tqdm
import numpy as np
import torch

def simulation(n, simulation_func):
    faces, total = simulation_func(n)

    print(f"\nRan {n} simulation{'s' if n >1 else ''}")
    print(f"Total Reward = {total}")
    print(faces)
def coin_toss_tensor(n):

    # create a sample of Bernoulli distribution for fair (50/50) coin
    # The size of the sample is n
    toss_results = torch.randint(0, 2, size=(n,), dtype=torch.int32)
    toss_results_list = toss_results.tolist()

    # return Counter object to summarize the counts of head/tail, and total rewards
    faces = Counter(toss_results_list)
    
    # Calculate total rewards
    total_reward = (toss_results * 4 - 2 * n).sum().item()
    return faces, total_reward


for n in [1, 1000, 1000000]:
    simulation(n, coin_toss_tensor)

# %%
