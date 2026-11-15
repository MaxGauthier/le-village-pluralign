"""Minimal pygame widgets: text, buttons, radio groups, steppers, bars, tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pygame

SURFACE = (252, 252, 251)
PANEL = (245, 245, 242)
INK = (11, 11, 11)
INK_2 = (82, 81, 78)
MUTED = (137, 135, 129)
GRID = (225, 224, 217)
AXIS = (195, 194, 183)
ACCENT = (42, 120, 214)
ACCENT_LIGHT = (205, 226, 251)
GOOD = (12, 163, 12)
BAD = (208, 59, 59)
WARN = (250, 178, 25)
WHITE = (255, 255, 255)

_FONTS: dict[tuple[int, bool], pygame.font.Font] = {}


def font(size: int = 13, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    if key not in _FONTS:
        _FONTS[key] = pygame.font.SysFont("segoeui,dejavusans,arial", size, bold=bold)
    return _FONTS[key]


def draw_text(surf, text, x, y, size=13, color=INK, bold=False, anchor="topleft") -> pygame.Rect:
    img = font(size, bold).render(str(text), True, color)
    rect = img.get_rect(**{anchor: (x, y)})
    surf.blit(img, rect)
    return rect


class Button:
    def __init__(self, rect, label, on_click: Callable | None = None, color=None, toggled=False,
                 enabled=True, size=13, sublabel: str | None = None, key=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.color = color
        self.toggled = toggled
        self.enabled = enabled
        self.size = size
        self.sublabel = sublabel
        self.key = key
        self.hover = False

    def draw(self, surf):
        if self.toggled:
            fill, border, ink = (self.color or ACCENT), (self.color or ACCENT), WHITE
        elif not self.enabled:
            fill, border, ink = PANEL, GRID, MUTED
        else:
            fill, border, ink = (ACCENT_LIGHT if self.hover else SURFACE), AXIS, INK
        pygame.draw.rect(surf, fill, self.rect, border_radius=6)
        pygame.draw.rect(surf, border, self.rect, width=1, border_radius=6)
        if self.color and not self.toggled:
            pygame.draw.rect(surf, self.color, (self.rect.x + 6, self.rect.y + 6, 6, self.rect.h - 12),
                             border_radius=3)
        if self.sublabel:
            draw_text(surf, self.label, self.rect.x + 16, self.rect.y + 4, self.size, ink, bold=False)
            draw_text(surf, self.sublabel, self.rect.x + 16, self.rect.y + 5 + self.size, self.size - 3,
                      WHITE if self.toggled else MUTED)
        else:
            draw_text(surf, self.label, self.rect.centerx, self.rect.centery, self.size, ink,
                      anchor="center")

    def handle(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.enabled \
                and self.rect.collidepoint(event.pos):
            if self.on_click:
                self.on_click()
            return True
        return False


class RadioGroup:
    """Vertical list of exclusive buttons."""

    def __init__(self, x, y, w, options: list[tuple[str, str, str | None]], selected, on_change,
                 row_h=30, colors: dict | None = None):
        self.selected = selected
        self.on_change = on_change
        self.buttons = []
        for i, (key, label, sub) in enumerate(options):
            b = Button((x, y + i * (row_h + 3), w, row_h), label, key=key, sublabel=sub, size=12,
                       color=(colors or {}).get(key))
            b.on_click = lambda k=key: self.select(k)
            self.buttons.append(b)
        self.height = len(options) * (row_h + 3)

    def select(self, key):
        self.selected = key
        self.on_change(key)

    def draw(self, surf):
        for b in self.buttons:
            b.toggled = b.key == self.selected
            b.draw(surf)

    def handle(self, event) -> bool:
        return any(b.handle(event) for b in self.buttons)


class Stepper:
    """< value > control over a discrete list of values."""

    def __init__(self, x, y, w, label, values, index, on_change, fmt=lambda v: f"{v:g}"):
        self.label = label
        self.values = list(values)
        self.index = index
        self.on_change = on_change
        self.fmt = fmt
        self.rect = pygame.Rect(x, y, w, 24)
        self.left = Button((x + w - 88, y, 24, 24), "<", lambda: self.step(-1), size=12)
        self.right = Button((x + w - 24, y, 24, 24), ">", lambda: self.step(1), size=12)

    @property
    def value(self):
        return self.values[self.index]

    def step(self, d):
        new = min(max(self.index + d, 0), len(self.values) - 1)
        if new != self.index:
            self.index = new
            self.on_change(self.value)

    def draw(self, surf):
        draw_text(surf, self.label, self.rect.x, self.rect.centery, 12, INK_2, anchor="midleft")
        draw_text(surf, self.fmt(self.value), self.rect.right - 44, self.rect.centery, 12, INK, True,
                  anchor="center")
        self.left.enabled = self.index > 0
        self.right.enabled = self.index < len(self.values) - 1
        self.left.draw(surf)
        self.right.draw(surf)

    def handle(self, event) -> bool:
        return self.left.handle(event) or self.right.handle(event)


def draw_bar(surf, x, y, w, h, value, vmin, vmax, color, marker=None, marker_color=INK):
    """Horizontal bar from zero; negative values extend left of the zero line."""
    pygame.draw.rect(surf, PANEL, (x, y, w, h), border_radius=3)
    span = max(vmax - vmin, 1e-9)

    def px(v):
        return x + int((min(max(v, vmin), vmax) - vmin) / span * w)

    zero = px(0.0)
    if value >= 0:
        pygame.draw.rect(surf, color, (zero, y, max(px(value) - zero, 1), h), border_radius=3)
    else:
        pygame.draw.rect(surf, color, (px(value), y, max(zero - px(value), 1), h), border_radius=3)
    pygame.draw.line(surf, AXIS, (zero, y - 2), (zero, y + h + 2), 1)
    if marker is not None:
        mx = px(marker)
        pygame.draw.polygon(surf, marker_color, [(mx - 4, y - 5), (mx + 4, y - 5), (mx, y)])


def draw_table(surf, x, y, col_widths, header, rows, row_h=19, highlight=None, size=11):
    cx = x
    for wcol, h in zip(col_widths, header):
        draw_text(surf, h, cx, y, size, INK_2, bold=True)
        cx += wcol
    pygame.draw.line(surf, AXIS, (x, y + row_h - 2), (x + sum(col_widths), y + row_h - 2), 1)
    for i, row in enumerate(rows):
        ry = y + (i + 1) * row_h
        if highlight is not None and i == highlight:
            pygame.draw.rect(surf, ACCENT_LIGHT, (x - 4, ry - 1, sum(col_widths) + 8, row_h), border_radius=3)
        cx = x
        for wcol, cell in zip(col_widths, row):
            text, color = (cell if isinstance(cell, tuple) else (cell, INK))
            draw_text(surf, text, cx, ry, size, color)
            cx += wcol
    return y + (len(rows) + 1) * row_h
