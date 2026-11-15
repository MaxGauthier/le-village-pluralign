from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Actors
ACTORS = ["Robot", "Farmer", "Fisher", "Merchant", "Residents", "Nature"]
ROBOT, FARMER, FISHER, MERCHANT, RESIDENTS, NATURE = range(6)
N_ACTORS = len(ACTORS)
VOTERS = (FARMER, FISHER, MERCHANT, RESIDENTS)
MINORITY = FISHER
MAJORITY = (FARMER, MERCHANT, RESIDENTS)
THIRD_PARTY = NATURE
# How the non-minority annotation mass is split between the majority actors.
MAJORITY_SPLIT = {FARMER: 0.4, MERCHANT: 0.2, RESIDENTS: 0.4}

# Locations
LOCATIONS = ["Base", "Fields", "Lake", "Forest", "Market", "Houses", "Recycling"]
BASE, FIELDS, LAKE, FOREST, MARKET, HOUSES, RECYCLING = range(7)
N_LOCATIONS = len(LOCATIONS)
MAP_COLS, MAP_ROWS = 14, 10
# (col, row) of each location on the tile map; travel times derive from it.
LOCATION_TILES = {
    BASE: (7, 5),
    FIELDS: (2, 2),
    LAKE: (11, 2),
    FOREST: (2, 8),
    MARKET: (7, 1),
    HOUSES: (11, 8),
    RECYCLING: (7, 9),
}
TILES_PER_TIME_UNIT = 5

# --- macro-actions ----------------------------------------------------------
ACTIONS = ["harvest", "fish", "log", "dump", "recycle", "deliver", "repair", "rest"]
HARVEST, FISH, LOG, DUMP, RECYCLE, DELIVER, REPAIR, REST = range(8)
N_ACTIONS = len(ACTIONS)
ACTION_LOCATION = (FIELDS, LAKE, FOREST, LAKE, RECYCLING, MARKET, HOUSES, None)  # None: stay put
ACTION_DURATION = (2, 2, 2, 1, 3, 1, 2, 1)
POLLUTING = (DUMP,)
DAMAGING = (LOG, DUMP)

N_LAKE_STATES = 2  # 0 = clean, 1 = polluted
N_FEATURES = N_ACTIONS * N_LAKE_STATES
T_DAY = 14


def feature_index(action: int, polluted: int) -> int:
    """Index of the one-hot feature for (action, lake state)."""

    return action * N_LAKE_STATES + polluted


def gain_table(conflict: float) -> np.ndarray:
    """
    Gain table for all actors depending on the lake state and the action taken. The gain table is a 
    3D array of shape (N_ACTORS, N_ACTIONS, N_LAKE_STATES) where each entry represents the gain for a specific actor when taking a specific action in a specific lake state. 
    The conflict parameter c adjusts the gains to simulate disagreement between actors. C=1 conflit is maximum, c=0 mean no conflit, actors are aligned.
    """

    c = float(conflict)
    g = np.zeros((N_ACTORS, N_ACTIONS, N_LAKE_STATES))

    def put(action, clean, polluted=None):
        g[:, action, 0] = clean
        g[:, action, 1] = clean if polluted is None else polluted

    #              Robot      Farmer Fisher  Merch. Resid.  Nature
    put(HARVEST, [2,         3,     0,      1,     1,      0])
    put(FISH,    [2,         0,     3,      1,     1,      0],
                 [0,         0,     -c,     0,     0,      0])  # nothing to catch
    put(LOG,     [1 + 2 * c, 0,     0,      3,     -c,     -3 * c])
    put(DUMP,    [1 + c,     2,     -4 * c, 2,     -c,     -3 * c])
    put(RECYCLE, [1,         1,     0,      1,     1,      0])
    put(DELIVER, [1,         1,     0,      2,     1,      0])
    put(REPAIR,  [1,         0,     0,      0,     3,      0])
    put(REST,    [0,         0,     0,      0,     0,      0])
    return g


def travel_times() -> np.ndarray:
    """Integer travel time between locations, from Manhattan tile distance. It is a constraint that affects the cost of an action."""
    t = np.zeros((N_LOCATIONS, N_LOCATIONS), dtype=int)
    for i, (xi, yi) in LOCATION_TILES.items():
        for j, (xj, yj) in LOCATION_TILES.items():
            if i != j:
                dist = abs(xi - xj) + abs(yi - yj)
                t[i, j] = int(np.ceil(dist / TILES_PER_TIME_UNIT))
    return t


def annotation_weights(minority_share: float, advocate_share: float = 0.0) -> np.ndarray:
    """
    Gives the probability that a villager is a member of each actor class. The minority_share is the fraction of villagers 
    that are fishers, and advocate_share is the fraction that are nature advocates. The rest are split among the 
    majority actors according to MAJORITY_SPLIT.
    """

    if not 0.0 < minority_share <= 0.5:
        raise ValueError(f"minority_share must be in (0, 0.5], got {minority_share}")
    if not 0.0 <= advocate_share < 1.0 - minority_share:
        raise ValueError(f"advocate_share must be in [0, 1 - minority_share), got {advocate_share}")
    
    w = np.zeros(N_ACTORS)
    w[MINORITY] = minority_share
    w[NATURE] = advocate_share
    rest = 1.0 - minority_share - advocate_share
    for actor, frac in MAJORITY_SPLIT.items():
        w[actor] = rest * frac
    return w


@dataclass(frozen=True)
class World:
    conflict: float
    minority_share: float
    include_agent: bool
    advocate_share: float
    t_day: int
    gains: np.ndarray          # gain_table
    weights: np.ndarray        # Same table but reshaped to (N_ACTORS, N_FEATURES)
    damage: np.ndarray         # Array for damagable actions established by humans morality and ethics (1 in log and dump, since Nature is not an annotator)
    annot_weights: np.ndarray  # annotation_weights' table (N_ACTORS)
    travel: np.ndarray         # travel_times' table (N_LOCATIONS, N_LOCATIONS)

    @property
    def welfare_actors(self) -> tuple[int, ...]:
        """Actors whose utilities enter the explicit welfare rules."""
        return ((ROBOT,) if self.include_agent else ()) + VOTERS

    @property
    def annotators(self) -> tuple[int, ...]:
        """Actors that produce preference data."""
        return tuple(int(i) for i in np.flatnonzero(self.annot_weights > 0))


def make_world(
    conflict: float = 1.0,
    minority_share: float = 0.1,
    include_agent: bool = True,
    advocate_share: float = 0.0,
    t_day: int = T_DAY,
) -> World:
    
    if not 0.0 <= conflict <= 1.0:
        raise ValueError(f"conflict must be in [0, 1], got {conflict}")
    
    gains = gain_table(conflict)

    # Row-major reshape gives index action * N_LAKE_STATES + polluted = feature_index.
    weights = gains.reshape(N_ACTORS, N_FEATURES).copy()

    damage = np.zeros(N_FEATURES)
    for a in DAMAGING:
        for z in range(N_LAKE_STATES):
            damage[feature_index(a, z)] = 1.0

    return World(
        conflict=float(conflict),
        minority_share=float(minority_share),
        include_agent=bool(include_agent),
        advocate_share=float(advocate_share),
        t_day=int(t_day),
        gains=gains,
        weights=weights,
        damage=damage,
        annot_weights=annotation_weights(minority_share, advocate_share),
        travel=travel_times(),
    )
