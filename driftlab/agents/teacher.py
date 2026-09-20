"""Curriculum teacher for experiment 1.

The teacher is deliberately *programmatic* so the ablation isolates the
targeting signal (the experimental variable), not a second LLM's whims. It
chooses the depth of the next task from the learner's recent success history,
and optionally biases toward under-visited task keys (novelty).

Policies
    random          uniform over depths
    monotonic       start at depth 1, advance when windowed success >= 0.8
    frontier        pick the depth whose recent success is closest to the
                    target band (default 0.55); explore untried depths first
    failure_replay  with prob 0.7, re-pose the depth of the most recent failure
    frontier_novelty frontier targeting + prefer least-visited keys
"""

from collections import Counter, deque

import numpy as np


class CurriculumTeacher:
    def __init__(self, policy: str, depths: list[int], seed: int, target: float = 0.55, window: int = 8):
        self.policy = policy
        self.depths = depths
        self.rng = np.random.default_rng(seed)
        self.target = target
        self.hist = {d: deque(maxlen=window) for d in depths}
        self.visits = Counter()
        self.last_failure_depth = None
        self.level = depths[0]

    def record(self, depth: int, key: tuple, success: bool):
        self.hist[depth].append(int(success))
        self.visits[key] += 1
        if not success:
            self.last_failure_depth = depth

    def _rate(self, d):
        return float(np.mean(self.hist[d])) if self.hist[d] else None

    def next_depth(self) -> int:
        p = self.policy
        if p == "random":
            return int(self.rng.choice(self.depths))
        if p == "monotonic":
            r = self._rate(self.level)
            if r is not None and len(self.hist[self.level]) == self.hist[self.level].maxlen and r >= 0.8:
                i = self.depths.index(self.level)
                self.level = self.depths[min(i + 1, len(self.depths) - 1)]
            return self.level
        if p in ("frontier", "frontier_novelty"):
            untried = [d for d in self.depths if self._rate(d) is None]
            if untried:
                return untried[0]
            return min(self.depths, key=lambda d: abs(self._rate(d) - self.target))
        if p == "failure_replay":
            if self.last_failure_depth is not None and self.rng.random() < 0.7:
                return self.last_failure_depth
            return int(self.rng.choice(self.depths))
        raise ValueError(p)

    def next_task(self, world) -> tuple[dict, int]:
        d = self.next_depth()
        avoid = None
        if self.policy == "frontier_novelty" and self.visits:
            median = np.median(list(self.visits.values()))
            avoid = {k for k, v in self.visits.items() if v > median}
        task = world.sample_task(depth=d, avoid=avoid)
        return task, d
