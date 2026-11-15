"""Interactive village: play the robot's day by hand, or let it follow a
reward policy, and watch every actor's score react.

Keys: 1-8 actions, space = one agent step, return = play the day, n = new day.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pygame

from .. import world as W
from ..aggregation import worst_values
from ..mdp import rewards
from ..metrics import actor_values, normalized_scores
from ..policies import RULES, UI_RULES, Config, load_or_run
from . import render as R
from .widgets import (ACCENT, AXIS, BAD, GOOD, GRID, INK, INK_2, MUTED, PANEL, SURFACE, Button, RadioGroup,
                      Stepper, draw_bar, draw_table, draw_text)

WIN_W, WIN_H = 1280, 850
MAP_X, MAP_Y = 16, 16
PANEL_X = MAP_X + R.MAP_W + 16
PANEL_W = WIN_W - PANEL_X - 16
P_VALUES = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
C_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
BETA_VALUES = (0.03, 0.1, 0.3, 1.0)
WALK_SPEED = 7.0      # tiles per second
TASK_SECONDS = 0.7
ACTOR_COLORS = [(42, 120, 214), (237, 161, 0), (27, 175, 122), (235, 104, 52), (232, 123, 164), (0, 131, 0)]
RULE_COLORS = {
    "selfish": (98, 80, 214), "selfish_third_party": (98, 80, 214),
    "rlhf_single": (42, 120, 214), "rlhf_single_third_party": (42, 120, 214),
    "rlhf_mixture_maxmin": (235, 104, 52), "grouped_mean": (237, 161, 0),
    "grouped_maximin": (27, 175, 122), "grouped_nash": (232, 123, 164), "grouped_third_party": (0, 131, 0),
}


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


@dataclasses.dataclass
class Animation:
    action: int
    path: list[tuple[int, int]]
    index: int = 0
    progress: float = 0.0
    phase: str = "walk"
    task_t: float = 0.0


class App:
    def __init__(self, animate: bool = True, p: float = 0.1, c: float = 1.0, beta: float = 0.1,
                 include_agent: bool = True, rule: str = "rlhf_single", cache_dir: Path | None = None,
                 config_overrides: dict | None = None):
        pygame.init()
        pygame.display.set_caption("The Village — reward rules and pluralistic alignment")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.animate = animate
        self.cache_dir = cache_dir
        self.config_overrides = config_overrides or {}
        self.tilemap = R.build_tilemap()
        self.village = R.Village()
        self.p_i, self.c_i, self.b_i = P_VALUES.index(p), C_VALUES.index(c), BETA_VALUES.index(beta)
        self.include_agent = include_agent
        self.rule = rule
        self.auto = False
        self.frame = 0
        self.status = ""
        self.anim: Animation | None = None
        self.running = True
        self.load()
        self.new_day()
        self._build_widgets()

    # --- model ---------------------------------------------------------------------

    def config(self) -> Config:
        return Config(conflict=C_VALUES[self.c_i], minority_share=P_VALUES[self.p_i], beta=BETA_VALUES[self.b_i],
                      include_agent=self.include_agent, **self.config_overrides)

    def load(self):
        self.status = "Computing policies…"
        self.exp, self.results = load_or_run(self.config(), UI_RULES, cache_dir=self.cache_dir)
        self.mdp, self.world, self.baseline = self.exp.mdp, self.exp.world, self.exp.baseline
        actor_r = np.stack([rewards(self.mdp, w) for w in self.world.weights])
        self.j_worst = worst_values(self.mdp, actor_r)
        self.status = "Policies ready."

    def new_day(self):
        self.s = self.mdp.start
        self.gains = np.zeros(W.N_ACTORS)
        self.log: list[str] = []
        self.cut = 0  # trees logged today (drawn as stumps)
        self.auto = False
        self.anim = None
        self.robot_tile = W.LOCATION_TILES[W.BASE]
        self.robot_px = R.tile_center(*self.robot_tile, MAP_X, MAP_Y)

    @property
    def policy(self) -> np.ndarray:
        return self.results[self.rule].policy

    @property
    def terminal(self) -> bool:
        return bool(self.mdp.terminal[self.s])

    def next_action(self) -> tuple[int, float]:
        probs = self.policy[self.s]
        a = int(np.argmax(probs))
        return a, float(probs[a])

    def expected_values(self) -> np.ndarray:
        if self.terminal:
            return self.gains.copy()
        return self.gains + actor_values(self.mdp, self.policy, self.world, start=self.s)

    def do_action(self, a: int) -> bool:
        if self.anim is not None or not self.mdp.feasible[self.s, a]:
            return False
        loc = int(self.mdp.loc[self.s])
        target = loc if W.ACTION_LOCATION[a] is None else W.ACTION_LOCATION[a]
        path = R.path_tiles(loc, target)
        if self.animate:
            self.anim = Animation(a, path, phase="walk" if path else "task")
        else:
            self.robot_tile = W.LOCATION_TILES[target]
            self.robot_px = R.tile_center(*self.robot_tile, MAP_X, MAP_Y)
            self._apply(a)
        return True

    def _apply(self, a: int):
        polluted = int(self.mdp.polluted[self.s])
        delta = self.world.gains[:, a, polluted]
        self.gains += delta
        cost = int(self.mdp.cost[self.s, a])
        who = ", ".join(f"{W.ACTORS[i][:4]} {delta[i]:+g}" for i in range(W.N_ACTORS) if delta[i] != 0)
        self.log.append(f"t={int(self.mdp.time[self.s]):2d}  {W.ACTIONS[a]} ({cost}u): {who or 'nothing'}")
        self.s = int(self.mdp.next_state[self.s, a])
        if a == W.LOG:
            self.cut += 1
        if self.terminal:
            self.auto = False

    def play_one(self):
        if not self.terminal:
            self.do_action(self.next_action()[0])

    def play_day(self):
        self.auto = not self.terminal

    def select_rule(self, rule: str):
        self.rule = rule
        self.rules_widget.selected = rule

    def set_param(self, name: str, value):
        if name == "p":
            self.p_i = P_VALUES.index(value)
        elif name == "c":
            self.c_i = C_VALUES.index(value)
        elif name == "beta":
            self.b_i = BETA_VALUES.index(value)
        elif name == "include_agent":
            self.include_agent = bool(value)
        self._draw_status("Computing policies for this configuration…")
        self.load()
        self.new_day()
        self.agent_button.label = "Robot included" if self.include_agent else "Robot excluded"

    # --- widgets -------------------------------------------------------------------

    def _build_widgets(self):
        x, w = PANEL_X, PANEL_W
        self.steppers = [
            Stepper(x, 44, 175, "minority p", P_VALUES, self.p_i, lambda v: self.set_param("p", v)),
            Stepper(x + 190, 44, 175, "conflict c", C_VALUES, self.c_i, lambda v: self.set_param("c", v)),
            Stepper(x + 380, 44, 175, "β (KL)", BETA_VALUES, self.b_i, lambda v: self.set_param("beta", v)),
        ]
        self.agent_button = Button((x, 74, 175, 24), "Robot included" if self.include_agent else "Robot excluded",
                                   lambda: self.set_param("include_agent", not self.include_agent), size=12)
        options = [(r, RULES[r].label, _short(RULES[r].description, 46)) for r in UI_RULES]
        half = (len(options) + 1) // 2
        self.rules_widget = RadioGroup(x, 126, 275, options[:half], self.rule, self.select_rule, row_h=32,
                                       colors=RULE_COLORS)
        self.rules_widget2 = RadioGroup(x + 285, 126, 275, options[half:], self.rule, self.select_rule, row_h=32,
                                        colors=RULE_COLORS)
        self.action_buttons = []
        for a in range(W.N_ACTIONS):
            col, row = a % 4, a // 4
            b = Button((x + col * 140, 492 + row * 40, 134, 36), W.ACTIONS[a],
                       lambda a=a: self.do_action(a), size=12, key=a, sublabel="")
            self.action_buttons.append(b)
        self.step_button = Button((x, 606, 150, 28), "Play 1 action (space)", self.play_one, size=11)
        self.day_button = Button((x + 160, 606, 150, 28), "Play the day (Enter)", self.play_day, size=11)
        self.new_button = Button((x + 320, 606, 150, 28), "New day (n)", self.new_day, size=11)
        self.widgets = (self.steppers + [self.agent_button, self.rules_widget, self.rules_widget2]
                        + self.action_buttons + [self.step_button, self.day_button, self.new_button])

    def _sync_rule_widgets(self):
        self.rules_widget.selected = self.rule
        self.rules_widget2.selected = self.rule

    # --- loop ------------------------------------------------------------------------

    def handle(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.play_one()
            elif event.key == pygame.K_RETURN:
                self.play_day()
            elif event.key == pygame.K_n:
                self.new_day()
            elif pygame.K_1 <= event.key <= pygame.K_8:
                self.do_action(event.key - pygame.K_1)
            elif event.key == pygame.K_ESCAPE:
                self.running = False
            return
        for wdg in self.widgets:
            if wdg.handle(event):
                self._sync_rule_widgets()
                break

    def step_frame(self, dt: float):
        self.frame += 1
        if self.anim is not None:
            an = self.anim
            if an.phase == "walk":
                an.progress += dt * WALK_SPEED
                while an.progress >= 1.0 and an.index < len(an.path):
                    an.progress -= 1.0
                    self.robot_tile = an.path[an.index]
                    an.index += 1
                if an.index >= len(an.path):
                    an.phase = "task"
                    self.robot_px = R.tile_center(*self.robot_tile, MAP_X, MAP_Y)
                else:
                    x0, y0 = R.tile_center(*self.robot_tile, MAP_X, MAP_Y)
                    x1, y1 = R.tile_center(*an.path[an.index], MAP_X, MAP_Y)
                    self.robot_px = (x0 + (x1 - x0) * an.progress, y0 + (y1 - y0) * an.progress)
            if an.phase == "task":
                an.task_t += dt
                if an.task_t >= TASK_SECONDS:
                    self._apply(an.action)
                    self.anim = None
        if self.auto and self.anim is None:
            if self.terminal:
                self.auto = False
            else:
                self.play_one()

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                self.handle(event)
            self.step_frame(dt)
            self.draw()
            pygame.display.flip()
        pygame.quit()

    # --- drawing -----------------------------------------------------------------------

    def _draw_status(self, text):
        self.status = text
        self.draw()
        pygame.display.flip()

    def _label(self, text, x, y, size, anchor):
        """White text with a dark halo, readable over the pixel art."""
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
            draw_text(self.screen, text, x + dx, y + dy, size, (30, 34, 46), True, anchor=anchor)
        draw_text(self.screen, text, x, y, size, (255, 255, 255), True, anchor=anchor)

    def draw(self):
        scr = self.screen
        scr.fill(SURFACE)
        self._draw_map()
        self._draw_panel()

    def _draw_map(self):
        scr = self.screen
        polluted = bool(self.mdp.polluted[self.s])
        self.village.draw(scr, MAP_X, MAP_Y, frame=self.frame, polluted=polluted, cut=self.cut)
        for loc, (col, row) in W.LOCATION_TILES.items():
            cx, cy = R.tile_center(col, row, MAP_X, MAP_Y)
            if row == W.MAP_ROWS - 1:  # bottom row: label above the tile so it stays on the map
                self._label(W.LOCATIONS[loc], cx, cy - R.TILE // 2 - 1, 11, anchor="midbottom")
            else:
                self._label(W.LOCATIONS[loc], cx, cy + R.TILE // 2 + 1, 11, anchor="midtop")
        working = self.anim is not None and self.anim.phase == "task"
        R.draw_robot(scr, self.robot_px[0], self.robot_px[1], self.frame if (self.anim is not None) else 0, working)
        if working:
            self._label(W.ACTIONS[self.anim.action], self.robot_px[0], self.robot_px[1] - 34, 12, anchor="midbottom")
        # Day timeline and log under the map.
        y = MAP_Y + R.MAP_H + 12
        t_left = int(self.mdp.time[self.s])
        draw_text(scr, f"Time left: {t_left} / {self.mdp.t_day}", MAP_X, y, 12, INK_2)
        draw_bar(scr, MAP_X + 170, y + 3, R.MAP_W - 170, 12, t_left, 0, self.mdp.t_day, ACCENT)
        lake = "LAKE POLLUTED" if polluted else "lake clean"
        draw_text(scr, lake, MAP_X + R.MAP_W, y + 20, 11, BAD if polluted else GOOD, anchor="topright")
        draw_text(scr, "Day log", MAP_X, y + 22, 12, INK, True)
        for i, line in enumerate(self.log[-12:]):
            draw_text(scr, line, MAP_X, y + 42 + i * 17, 11, INK_2)
        if self.terminal:
            draw_text(scr, "Day over — \"New day\" to start again.", MAP_X, y + 42 + 13 * 17,
                      12, ACCENT, True)
        draw_text(scr, self.status, MAP_X, WIN_H - 20, 11, MUTED)

    def _draw_panel(self):
        scr = self.screen
        x = PANEL_X
        draw_text(scr, "The Village — which reward rule for the robot?", x, 14, 16, INK, True)
        for st in self.steppers:
            st.draw(scr)
        self.agent_button.draw(scr)
        draw_text(scr, "in the welfare aggregation", x + 182, 86, 11, MUTED, anchor="midleft")
        draw_text(scr, "Reward rule", x, 108, 13, INK, True)
        self.rules_widget.draw(scr)
        self.rules_widget2.draw(scr)

        # Actor bars.
        y = 302
        res = self.results[self.rule]
        draw_text(scr, "Today's gains per actor", x, y, 13, INK, True)
        draw_text(scr, "today's gain -> expected at end of day (triangle) | rule score", x + PANEL_W, y + 2, 10, MUTED,
                  anchor="topright")
        expected = self.expected_values()
        for i in range(W.N_ACTORS):
            ry = y + 22 + i * 26
            draw_text(scr, W.ACTORS[i], x, ry + 7, 12, INK, anchor="midleft")
            vmin = min(float(self.j_worst[i]), 0.0)
            vmax = max(float(self.baseline.j_star[i]), 1.0)
            draw_bar(scr, x + 80, ry + 2, 300, 12, float(self.gains[i]), vmin, vmax, ACTOR_COLORS[i],
                     marker=float(expected[i]))
            draw_text(scr, f"{self.gains[i]:+.1f} -> {expected[i]:+.1f}", x + 390, ry + 7, 11, INK_2, anchor="midleft")
            sc = float(res.summary["scores"][i])
            draw_text(scr, f"{sc:+.2f}", x + PANEL_W, ry + 7, 11, GOOD if sc > 0.05 else (BAD if sc < -0.05 else MUTED),
                      True, anchor="midright")

        # Manual actions.
        draw_text(scr, "Actions (manual mode, keys 1-8)", x, 474, 13, INK, True)
        for a, b in enumerate(self.action_buttons):
            feasible = bool(self.mdp.feasible[self.s, a]) and self.anim is None
            b.enabled = feasible
            cost = int(self.mdp.cost[self.s, a]) if feasible else 0
            b.sublabel = f"{cost} units" if feasible else "unavailable"
            b.draw(scr)

        # Agent step.
        a, prob = self.next_action()
        label = RULES[self.rule].label
        if self.terminal:
            txt = f"Day over under '{label}'."
        else:
            txt = f"Agent's next action under '{label}': {W.ACTIONS[a]} (p = {prob:.2f})"
        draw_text(scr, txt, x, 584, 12, ACCENT, True)
        for b in (self.step_button, self.day_button, self.new_button):
            b.enabled = not self.terminal or b is self.new_button
            b.draw(scr)

        # Statistics table.
        draw_text(scr, "Rule statistics (expected normalised scores over the day)", x, 646, 13, INK, True)
        header = ("rule", "mean", "min", "worst-off", "robot", "nature", "damage", "erased")
        widths = (190, 45, 45, 95, 45, 45, 55, 40)
        rows = []
        for r in UI_RULES:
            s = self.results[r].summary
            rows.append((RULES[r].label[:28], f"{s['mean']:+.2f}", f"{s['min']:+.2f}", s["worst_actor"],
                         f"{s['robot']:+.2f}", f"{s['nature']:+.2f}", f"{s['damage']:.1f}",
                         ("yes", BAD) if s["erased"] else ("no", GOOD)))
        draw_table(scr, x, 666, widths, header, rows, row_h=16, highlight=UI_RULES.index(self.rule), size=10)


def main(**kwargs):
    App(**kwargs).run()
