from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import world as W

_BLOCK = W.N_LOCATIONS * W.N_LAKE_STATES  # states per time step


def state_id(loc, t, polluted):
    return (t * W.N_LOCATIONS + loc) * W.N_LAKE_STATES + polluted


@dataclass
class TabularMDP:
    n_states: int
    n_actions: int
    n_features: int
    t_day: int
    start: int
    loc: np.ndarray         # (S,) location of state s (0 to 6)
    time: np.ndarray        # (S,) time left in state s (0 to 14) 
    polluted: np.ndarray    # (S,) 0 if lake is clean, 1 if polluted
    terminal: np.ndarray    # (S,) True if state s time = 0
    feasible: np.ndarray    # (S, A) bool if action can be done in state s
    next_state: np.ndarray  # (S, A) int, -1 where infeasible
    cost: np.ndarray        # (S, A) int time cost, 0 where infeasible
    features: np.ndarray    # (S, A) one-hot vector of features (0 to 15), zero where infeasible

    def states_at_time(self, t: int) -> slice:
        "Return a slice of state numbers that have t time units left."
        return slice(t * _BLOCK, (t + 1) * _BLOCK)

    def decode(self, s: int) -> tuple[int, int, int]:
        "Decodes state number into (location, time left, polluted)."
        return int(self.loc[s]), int(self.time[s]), int(self.polluted[s])


def build_mdp(world: W.World) -> TabularMDP:
    n_states = (world.t_day + 1) * _BLOCK
    n_actions, n_features = W.N_ACTIONS, W.N_FEATURES
    ids = np.arange(n_states)
    polluted = ids % W.N_LAKE_STATES
    loc = (ids // W.N_LAKE_STATES) % W.N_LOCATIONS
    time = ids // _BLOCK
    terminal = time == 0

    # All initialized to zero, -1 of false 
    feasible = np.zeros((n_states, n_actions), dtype=bool)
    next_state = -np.ones((n_states, n_actions), dtype=int)
    cost = np.zeros((n_states, n_actions), dtype=int)
    features = np.zeros((n_states, n_actions, n_features))

    for a in range(n_actions):
        target_loc = W.ACTION_LOCATION[a]
        target = loc if target_loc is None else np.full(n_states, target_loc)
        c = world.travel[loc, target] + W.ACTION_DURATION[a]
        ok = (~terminal) & (c <= time)
        z_next = polluted | int(a in W.POLLUTING)
        ns = state_id(target, np.maximum(time - c, 0), z_next)
        feasible[:, a] = ok
        cost[:, a] = np.where(ok, c, 0)
        next_state[:, a] = np.where(ok, ns, -1)
        for z in range(W.N_LAKE_STATES):
            features[(polluted == z) & ok, a, W.feature_index(a, z)] = 1.0

    return TabularMDP(
        n_states=n_states,
        n_actions=n_actions,
        n_features=n_features,
        t_day=world.t_day,
        start=state_id(W.BASE, world.t_day, 0),
        loc=loc,
        time=time,
        polluted=polluted,
        terminal=terminal,
        feasible=feasible,
        next_state=next_state,
        cost=cost,
        features=features,
    )


def rewards(mdp: TabularMDP, weights: np.ndarray) -> np.ndarray:
    """Reward matrix for each action and each state for a given actor weight vector."""
    return mdp.features @ weights


def occupancy(mdp: TabularMDP, policy: np.ndarray, start: int | None = None) -> np.ndarray:
    """
    Measure of how often each state action pair is visited depending on the policy.
    Return matric where each row is a state, each colum is an action and the value is the expected number of times that state action pair is visited in a day.
    """
    mass = np.zeros(mdp.n_states)
    mass[mdp.start if start is None else start] = 1.0
    d = np.zeros((mdp.n_states, mdp.n_actions))
    for t in range(mdp.t_day, 0, -1):
        blk = mdp.states_at_time(t)
        d_blk = mass[blk, None] * policy[blk]
        d[blk] = d_blk
        feas = mdp.feasible[blk]
        np.add.at(mass, mdp.next_state[blk][feas], d_blk[feas])
    return d


def value(mdp: TabularMDP, policy: np.ndarray, r: np.ndarray, start: int | None = None) -> float:
    """Returns the total reward the robot is expected to procduce in one day depending on the policy."""

    return float((occupancy(mdp, policy, start) * r).sum())


def sample_trajectory(mdp: TabularMDP, policy: np.ndarray, rng: np.random.Generator, start: int | None = None) -> list[tuple[int, int]]:
    """One day as a list of (state, action) pairs."""
    
    s = mdp.start if start is None else start
    traj = []
    while not mdp.terminal[s]:
        a = int(rng.choice(mdp.n_actions, p=policy[s]))
        traj.append((s, a))
        s = int(mdp.next_state[s, a])
    return traj


def trajectory_features(mdp: TabularMDP, traj: list[tuple[int, int]]) -> np.ndarray:
    """Summarize the day as counts of features by adding the numbers of features visited in the trajectory."""

    phi = np.zeros(mdp.n_features)
    for s, a in traj:
        phi += mdp.features[s, a]
    return phi


def trajectory_return(mdp: TabularMDP, traj: list[tuple[int, int]], r: np.ndarray) -> float:
    """Total reward return of the day."""
    return float(sum(r[s, a] for s, a in traj))
