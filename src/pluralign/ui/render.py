"""Bird's-eye pixel-art rendering of the village.

Ground, trees, houses and villagers come from the CC0 « 16x16 Puny World »
tileset by Shade (OpenGameArt, see ``assets/ATTRIBUTION.md``): 16x16 cells
scaled x3, autotiled (grass, dirt paths, sand, animated water, forest).  The
robot stays a small procedural sprite so that it is visually distinct from the
villagers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame

from .. import world as W

PIX = 3                   # screen pixels per art pixel
TILE = 16 * PIX           # 48 px
COLS, ROWS = W.MAP_COLS, W.MAP_ROWS
MAP_W, MAP_H = COLS * TILE, ROWS * TILE
ASSET_DIR = Path(__file__).resolve().parent / "assets"
SHEET_PATH = ASSET_DIR / "punyworld-overworld-tileset.png"

# --- logical tile map (used by the model-side tests and the travel paths) -------------
GRASS, PATH, WATER, FIELD, FOREST, MARKET, HOUSE, RECYCLE, BASE = range(9)
KIND_NAMES = ["grass", "path", "water", "field", "forest", "market", "house", "recycle", "base"]
# Zones (x0, y0, x1, y1), end exclusive, around the location tiles of world.py.
ZONES = {
    W.LAKE: (9, 0, 14, 5, WATER),
    W.FIELDS: (0, 0, 5, 5, FIELD),
    W.FOREST: (0, 6, 5, 10, FOREST),
    W.MARKET: (6, 0, 9, 3, MARKET),
    W.HOUSES: (9, 6, 14, 10, HOUSE),
    W.RECYCLING: (6, 8, 9, 10, RECYCLE),
    W.BASE: (6, 4, 9, 7, BASE),
}


def build_tilemap() -> np.ndarray:
    tm = np.full((ROWS, COLS), GRASS, dtype=int)
    tm[5, :] = PATH
    tm[:, 7] = PATH
    tm[:, 5] = np.where(tm[:, 5] == GRASS, PATH, tm[:, 5])
    for x0, y0, x1, y1, kind in ZONES.values():
        tm[y0:y1, x0:x1] = kind
    return tm


def tile_center(col: int, row: int, ox: int = 0, oy: int = 0) -> tuple[int, int]:
    return ox + col * TILE + TILE // 2, oy + row * TILE + TILE // 2


def path_tiles(src: int, dst: int) -> list[tuple[int, int]]:
    """Manhattan walk between two locations: horizontal first, then vertical."""
    (x0, y0), (x1, y1) = W.LOCATION_TILES[src], W.LOCATION_TILES[dst]
    tiles = []
    x, y = x0, y0
    while x != x1:
        x += 1 if x1 > x else -1
        tiles.append((x, y))
    while y != y1:
        y += 1 if y1 > y else -1
        tiles.append((x, y))
    return tiles


# --- visual design --------------------------------------------------------------------
# Ground letters: G grass, P dirt path, S sand, W water, C crop soil, F forest.
# Same layout as the logical map: fields top-left, lake top-right, forest
# bottom-left, houses bottom-right, market / base / recycling down the middle.
DESIGN = [
    "CCCCCPGGGGWWWG",
    "CCCCCPGPGWWWWW",
    "CCCCCPGPGWWWWW",
    "CCCCCPGPSWWWWW",
    "CCCCCPGPSSSSSS",
    "PPPPPPPPPPPPPP",
    "FFGFFPGPGGGPGG",
    "FFGFFPGPGGGPGG",
    "FFGFFPGPGGGPGG",
    "FFFFFPGPGGGGGG",
]
assert len(DESIGN) == ROWS and all(len(row) == COLS for row in DESIGN)

# Sheet cells (col, row) of 16x16 tiles.
GRASS_PLAIN = (0, 0)
GRASS_TUFTS = [(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
STUMP, WHEAT, HAY, BUSH, ROCK, PINE, ROUND_TREE = (1, 27), (0, 28), (1, 28), (0, 26), (1, 26), (0, 29), (0, 27)
SIGNPOST, CHEST_OPEN, CHEST = (7, 30), (2, 29), (3, 29)
STALL_A, STALL_B, STALL_C, STALL_GOODS = (4, 31), (5, 31), (8, 31), (9, 31)
GATE_TOP, GATE, TOWER_TOP, TOWER_BASE = (11, 26), (11, 27), (10, 26), (10, 27)
HALL_TOP, HALL_BOTTOM = [(12, 26), (13, 26), (12, 26, True)], [(12, 27), (13, 27), (12, 27, True)]  # True: mirrored
PIER_END, PIER = (10, 30), (10, 32)
HOUSE_ORANGE_A, HOUSE_ORANGE_B, HOUSE_ORANGE_C = (7, 26), (8, 26), (9, 26)
HOUSE_TEAL_A, HOUSE_TEAL_B, HOUSE_RED_A, HOUSE_RED_C = (7, 33), (8, 33), (17, 33), (19, 33)
HUT_ROUND, HUT_TENT = (6, 26), (4, 26)
NPC_ORANGE = [(4, 30), (5, 30), (6, 30)]
NPC_TEAL = [(4, 37), (5, 37), (6, 37)]
NPC_RED = [(14, 37), (15, 37), (16, 37)]

# Static objects drawn over the ground: (col, row, sheet cell).
OBJECTS = [
    # market: stalls around the robot's spot (7, 1), merchant above it
    (6, 0, SIGNPOST), (8, 0, CHEST_OPEN),
    (6, 1, STALL_A), (8, 1, STALL_GOODS), (6, 2, STALL_B),
    # robot depot: orange-gabled hall whose gate (7, 4) opens on the crossroads (7, 5)
    *[(6 + i, 3, cell) for i, cell in enumerate(HALL_TOP)], *[(6 + i, 4, cell) for i, cell in enumerate(HALL_BOTTOM)],
    # recycling plant: stone gatehouse, the robot stands in its gate (7, 9)
    (6, 8, TOWER_TOP), (7, 8, GATE_TOP), (8, 8, TOWER_TOP), (6, 9, TOWER_BASE), (7, 9, GATE), (8, 9, TOWER_BASE),
    # pier into the lake, ending on the fishing tile (11, 2)
    (11, 2, PIER_END), (11, 3, PIER),
    # houses along a small lane (col 11), robot stops at (11, 8)
    (9, 6, HOUSE_ORANGE_A), (10, 6, HOUSE_TEAL_B), (12, 6, HOUSE_RED_C), (13, 6, HOUSE_ORANGE_C),
    (9, 8, HOUSE_RED_A), (13, 8, HOUSE_TEAL_A), (10, 9, HUT_ROUND), (12, 9, HUT_TENT),
    # scenery
    (13, 0, PINE), (9, 0, BUSH), (8, 7, ROCK), (9, 7, ROUND_TREE), (13, 9, PINE),
    (11, 9, ROCK), (12, 4, CHEST), (13, 4, ROCK), (6, 7, BUSH), (8, 2, BUSH),
    # hay bales on the field lane
    (0, 2, HAY), (4, 2, HAY),
]
# Wheat on every crop cell except the lane (row 2) where the farmer and the robot stand.
OBJECTS += [(c, r, WHEAT) for r in (0, 1, 3, 4) for c in range(5)]

# Villagers: (col, row, sheet cell, bob phase).  Drawn every frame with a 1-px idle bob.
VILLAGERS = [
    (3, 2, NPC_ORANGE[1], 0),   # farmer on the field lane
    (10, 4, NPC_TEAL[0], 1),    # fisher on the beach by the pier
    (7, 0, NPC_RED[0], 0),      # merchant behind the stalls
    (10, 8, NPC_ORANGE[2], 1),  # residents
    (12, 8, NPC_TEAL[1], 0),
]

# Forest cells replaced by a stump after each logging action of the day.
CUT_ORDER = [(1, 7), (3, 7), (1, 8), (3, 8), (0, 6), (4, 6), (1, 6), (3, 6), (0, 7), (4, 7),
             (0, 8), (4, 8), (1, 9), (3, 9), (2, 9), (0, 9), (4, 9)]

# --- autotiling ---------------------------------------------------------------------
_CARD = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
_DIAG = {"NW": (-1, -1), "NE": (1, -1), "SW": (-1, 1), "SE": (1, 1)}
_EDGE_KEY = {
    frozenset(): "C",
    frozenset("N"): "N", frozenset("S"): "S", frozenset("E"): "E", frozenset("W"): "W",
    frozenset("NW"): "NW", frozenset("NE"): "NE", frozenset("SW"): "SW", frozenset("SE"): "SE",
    frozenset("NS"): "H", frozenset("EW"): "V",
    frozenset("NSW"): "HL", frozenset("NSE"): "HR", frozenset("NEW"): "VT", frozenset("SEW"): "VB",
    frozenset("NSEW"): "O",
}
_INNER_KEY = {
    frozenset({"NW"}): "iNW", frozenset({"NE"}): "iNE", frozenset({"SW"}): "iSW", frozenset({"SE"}): "iSE",
    frozenset({"NW", "NE"}): "iN", frozenset({"SW", "SE"}): "iS",
    frozenset({"NW", "SW"}): "iW", frozenset({"NE", "SE"}): "iE",
    frozenset({"NW", "SE"}): "iD1", frozenset({"NE", "SW"}): "iD2",
}


def autotile_key(same) -> str:
    """Blob-autotile key for a cell given ``same(dx, dy) -> bool`` on its 8 neighbours.

    Keys: C centre; N/S/E/W edges; NW/NE/SW/SE outer corners; H/V strips with
    their ends HL/HR/VT/VB; O isolated; i* inner corners (missing diagonals).
    """
    missing = frozenset(k for k, (dx, dy) in _CARD.items() if not same(dx, dy))
    if missing:
        return _EDGE_KEY[missing]
    miss_d = frozenset(k for k, (dx, dy) in _DIAG.items() if not same(dx, dy))
    if not miss_d:
        return "C"
    return _INNER_KEY.get(miss_d, "iX")


def _blob(c0, r0):
    return {"NW": (c0, r0), "N": (c0 + 1, r0), "NE": (c0 + 2, r0),
            "W": (c0, r0 + 1), "C": (c0 + 1, r0 + 1), "E": (c0 + 2, r0 + 1),
            "SW": (c0, r0 + 2), "S": (c0 + 1, r0 + 2), "SE": (c0 + 2, r0 + 2)}


def _plus(c, r):  # inner-corner pairs around the 4-corner centre (c, r)
    return {"iN": (c, r - 1), "iW": (c - 1, r), "iX": (c, r), "iE": (c + 1, r), "iS": (c, r + 1)}


def _quad(c0, r0):  # single inner corners
    return {"iNW": (c0, r0), "iNE": (c0 + 1, r0), "iSW": (c0, r0 + 1), "iSE": (c0 + 1, r0 + 1)}


def _dirt_table(off: int, plain: bool = False) -> dict:
    t = _blob(10 + off, 0) if plain else _blob(4 + off, 0)
    t.update(_plus(8 + off, 1))
    t.update(_quad(13 + off, 0))
    t.update({"iD2": (13 + off, 2), "iD1": (14 + off, 2),
              "VT": (3 + off, 0), "V": (3 + off, 1), "VB": (3 + off, 2), "O": (3 + off, 3),
              "HL": (4 + off, 3), "H": (5 + off, 3), "HR": (6 + off, 3)})
    return t


DIRT = _dirt_table(0)
FIELD_SOIL = _dirt_table(0, plain=True)
SAND = _dirt_table(12)
N_WATER_FRAMES = 4


def _water_table(f: int) -> dict:  # grass shore, animation frame f
    t = _blob(7, 10 + 3 * f)
    t.update(_plus(5, 11 + 3 * f))
    t.update(_quad(10, 10 + 2 * f))
    r = 10 + 4 * f
    t.update({"VT": (0, r), "V": (0, r + 1), "VB": (0, r + 2), "O": (0, r + 3),
              "HL": (1, r + 3), "H": (2, r + 3), "HR": (3, r + 3)})
    return t


def _water_sand_table(f: int) -> dict:  # sandy shore
    t = _blob(12, 10 + 3 * f)
    t.update(_quad(15, 10 + 2 * f))
    return t


WATER = [_water_table(f) for f in range(N_WATER_FRAMES)]
WATER_SAND = [_water_sand_table(f) for f in range(N_WATER_FRAMES)]
FOREST_T = _blob(0, 7)
FOREST_T.update({"O": (8, 8), "VT": (8, 7), "V": (8, 8), "VB": (8, 9), "HL": (8, 7), "H": (8, 8), "HR": (8, 9)})


def pick(table: dict, key: str) -> tuple[int, int]:
    if key in table:
        return table[key]
    if key.startswith("i") and "iX" in table:
        return table["iX"]
    return table["C"]


# --- sprite sheet -------------------------------------------------------------------
class Sheet:
    def __init__(self, path: Path = SHEET_PATH):
        img = pygame.image.load(str(path))
        try:
            img = img.convert_alpha()
        except pygame.error:  # no display yet
            pass
        self.img = img
        self._cache: dict[tuple[int, int], pygame.Surface] = {}

    def tile(self, c: int, r: int, flip: bool = False) -> pygame.Surface:
        key = (c, r, flip)
        if key not in self._cache:
            sub = self.img.subsurface((c * 16, r * 16, 16, 16))
            if flip:
                sub = pygame.transform.flip(sub, True, False)
            self._cache[key] = pygame.transform.scale(sub, (TILE, TILE))
        return self._cache[key]


def _grass_variant(c: int, r: int) -> tuple[int, int]:
    h = (c * 7919 + r * 104729) % 23
    return GRASS_TUFTS[h % len(GRASS_TUFTS)] if h < 6 else GRASS_PLAIN


class Village:
    """Baked map layers (per water frame, pollution state and number of cut trees)."""

    def __init__(self, sheet: Sheet | None = None):
        self.sheet = sheet or Sheet()
        self._layers: dict[tuple, pygame.Surface] = {}
        self._sludge: pygame.Surface | None = None

    # -- public --------------------------------------------------------------------
    def draw(self, surf: pygame.Surface, ox: int, oy: int, frame: int = 0, polluted: bool = False,
             cut: int = 0):
        wf = (frame // 12) % N_WATER_FRAMES
        surf.blit(self.layer(wf, polluted, cut), (ox, oy))
        for c, r, cell, phase in VILLAGERS:
            bob = PIX if ((frame // 24) + phase) % 2 else 0
            surf.blit(self.sheet.tile(*cell), (ox + c * TILE, oy + r * TILE - bob))

    def layer(self, wf: int, polluted: bool, cut: int) -> pygame.Surface:
        key = (wf, bool(polluted), min(int(cut), len(CUT_ORDER)))
        if key not in self._layers:
            if len(self._layers) >= 24:  # bound memory: ~1.3 MB per baked layer
                self._layers.clear()
            self._layers[key] = self._build(*key)
        return self._layers[key]

    # -- building ------------------------------------------------------------------
    def _build(self, wf: int, polluted: bool, cut: int) -> pygame.Surface:
        design = [list(row) for row in DESIGN]
        stumps = CUT_ORDER[:cut]
        for c, r in stumps:
            design[r][c] = "G"

        def at(c, r, default):
            return design[r][c] if 0 <= c < COLS and 0 <= r < ROWS else default

        surf = pygame.Surface((MAP_W, MAP_H))
        tile = self.sheet.tile
        water_cells = []
        for r in range(ROWS):
            for c in range(COLS):
                letter = design[r][c]
                pos = (c * TILE, r * TILE)
                surf.blit(tile(*_grass_variant(c, r)), pos)
                if letter in "PSC":
                    # Water draws its own shore, so it counts as "same" here.
                    key = autotile_key(lambda dx, dy: at(c + dx, r + dy, "P") in "PSCW")
                    table = {"P": DIRT, "S": SAND, "C": FIELD_SOIL}[letter]
                    surf.blit(tile(*pick(table, key)), pos)
                elif letter == "W":
                    key = autotile_key(lambda dx, dy: at(c + dx, r + dy, "W") == "W")
                    beach = any(at(c + dx, r + dy, "W") in "PS" for dx, dy in _CARD.values())
                    table = WATER_SAND[wf] if beach else WATER[wf]
                    surf.blit(tile(*pick(table, key)), pos)
                    water_cells.append(pos)
        if polluted:
            self._pollute(surf, water_cells)
        # Trees (drawn after the ground so that canopies overlap neighbours).
        for r in range(ROWS):
            for c in range(COLS):
                if design[r][c] == "F":
                    key = autotile_key(lambda dx, dy: at(c + dx, r + dy, "F") == "F")
                    surf.blit(tile(*pick(FOREST_T, key)), (c * TILE, r * TILE))
        objects = OBJECTS + [(c, r, STUMP) for c, r in stumps]
        for c, r, cell in sorted(objects, key=lambda o: (o[1], o[0])):
            surf.blit(tile(*cell), (c * TILE, r * TILE))
        return surf

    def _pollute(self, surf: pygame.Surface, cells: list[tuple[int, int]]):
        """Turn the bluish pixels of the lake murky green-brown and add sludge."""
        arr = pygame.surfarray.pixels3d(surf)
        for x, y in cells:
            sub = arr[x:x + TILE, y:y + TILE].astype(np.int16)
            mask = sub[..., 2] > sub[..., 0] + 25
            tinted = sub * np.array([0.45, 0.55, 0.25]) + np.array([70, 45, 0])
            arr[x:x + TILE, y:y + TILE] = np.clip(np.where(mask[..., None], tinted, sub), 0, 255).astype(np.uint8)
        del arr
        if self._sludge is None:
            self._sludge = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
            rng = np.random.default_rng(3)
            for _ in range(9):
                cx = rng.uniform(9.6, 13.2) * TILE
                cy = rng.uniform(0.4, 3.4) * TILE
                w, h = rng.uniform(0.5, 1.1) * TILE, rng.uniform(0.3, 0.6) * TILE
                pygame.draw.ellipse(self._sludge, (58, 48, 18, 120), (cx - w / 2, cy - h / 2, w, h))
        surf.blit(self._sludge, (0, 0))


# --- robot sprite -------------------------------------------------------------------
C = {
    "robot": (96, 116, 150), "robot2": (178, 194, 220), "tread": (58, 64, 80),
    "eye": (255, 230, 80), "antenna": (224, 64, 64), "lamp": (120, 220, 140), "outline": (28, 32, 46),
}
_ROBOT_CACHE: dict[int, pygame.Surface] = {}
_SHADOW: pygame.Surface | None = None


def _robot_surface(step: int) -> pygame.Surface:
    """16x18 art pixels with a 1-px dark outline, scaled x3."""
    if step in _ROBOT_CACHE:
        return _ROBOT_CACHE[step]
    art = pygame.Surface((16, 18), pygame.SRCALPHA)

    def rect(c0, r, w, h, key):
        pygame.draw.rect(art, C[key], (c0 + 1, r + 1, w, h))

    rect(6, 0, 2, 3, "antenna")
    rect(3, 3, 8, 6, "robot2")
    rect(4, 4, 6, 4, "robot")
    rect(5, 5, 1, 1, "eye")
    rect(8, 5, 1, 1, "eye")
    rect(2, 9, 10, 5, "robot")
    rect(6, 10, 2, 1, "lamp")
    rect(0, 10, 2, 3, "robot2")
    rect(12, 10, 2, 3, "robot2")
    rect(3 + step, 14, 3, 2, "tread")
    rect(8 - step, 14, 3, 2, "tread")
    alpha = pygame.surfarray.pixels_alpha(art)
    mask = alpha > 0
    grown = mask.copy()
    grown[1:, :] |= mask[:-1, :]
    grown[:-1, :] |= mask[1:, :]
    grown[:, 1:] |= mask[:, :-1]
    grown[:, :-1] |= mask[:, 1:]
    outline = grown & ~mask
    rgb = pygame.surfarray.pixels3d(art)
    rgb[outline] = C["outline"]
    alpha[outline] = 255
    del rgb, alpha
    _ROBOT_CACHE[step] = pygame.transform.scale(art, (16 * PIX, 18 * PIX))
    return _ROBOT_CACHE[step]


def draw_robot(surf, px: float, py: float, frame: int = 0, working: bool = False):
    """Robot sprite centred on (px, py), with a drop shadow."""
    global _SHADOW
    if _SHADOW is None:
        _SHADOW = pygame.Surface((12 * PIX, 4 * PIX), pygame.SRCALPHA)
        pygame.draw.ellipse(_SHADOW, (0, 0, 0, 70), _SHADOW.get_rect())
    bob = (frame // 8) % 2 if working else 0
    step = (frame // 6) % 2
    surf.blit(_SHADOW, (int(px) - 6 * PIX, int(py) + 5 * PIX))
    surf.blit(_robot_surface(step), (int(px) - 8 * PIX, int(py) - 9 * PIX - bob * PIX))
