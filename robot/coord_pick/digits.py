"""숫자 0~9 → 5행×3열 칸(3cm 블록 하나 = 칸 하나). 위 행이 0. '#' = 블록.

SMALL = 입회 시험용 3행 글꼴(1 = 블록 3개, 7 = 블록 5개).
"""
from __future__ import annotations

FONT = {
    "0": ("###", "#.#", "#.#", "#.#", "###"),
    "1": (".#.", "##.", ".#.", ".#.", "###"),
    "2": ("###", "..#", "###", "#..", "###"),
    "3": ("###", "..#", "###", "..#", "###"),
    "4": ("#.#", "#.#", "###", "..#", "..#"),
    "5": ("###", "#..", "###", "..#", "###"),
    "6": ("###", "#..", "###", "#.#", "###"),
    "7": ("###", "..#", "..#", "..#", "..#"),
    "8": ("###", "#.#", "###", "#.#", "###"),
    "9": ("###", "#.#", "###", "..#", "###"),
}
SMALL = {
    "1": (".#.", ".#.", ".#."),
    "7": ("###", "..#", "..#"),
}
ROWS, COLS = 5, 3


def pattern(digit: str | int, small: bool = False) -> tuple[str, ...]:
    d = str(digit)
    if small and d not in SMALL:
        raise ValueError(f"작은 글꼴은 {', '.join(SMALL)} 만 있다")
    return SMALL[d] if small else FONT[d]


def cells(digit: str | int, small: bool = False) -> list[tuple[int, int]]:
    """(행, 열) 목록. 행 0 = 위(사람이 숫자를 읽는 방향의 위)."""
    pat = pattern(digit, small)
    return [(r, c) for r in range(len(pat)) for c in range(COLS) if pat[r][c] == "#"]


def table() -> str:
    """사람이 보는 표: 숫자마다 그림과 블록 개수."""
    lines = ["  " + "   ".join(f"{d}({len(cells(d)):>2})" for d in FONT)]
    for r in range(ROWS):
        lines.append("  " + "   ".join(f"{FONT[d][r].replace('#', '■').replace('.', '·'):<5}" for d in FONT))
    return "\n".join(lines)
