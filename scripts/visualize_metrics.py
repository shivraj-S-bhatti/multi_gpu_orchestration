#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render training metrics JSONL as a standalone SVG.")
    parser.add_argument("input", type=Path, help="Path to a JSONL metrics file.")
    parser.add_argument("output", type=Path, help="Where to write the SVG.")
    parser.add_argument(
        "--title",
        type=str,
        default="Training Metrics",
        help="Chart title shown at the top of the SVG.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No metrics rows found in {path}.")
    return rows


def unique_by_step(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    deduped: dict[int, dict[str, float]] = {}
    for row in rows:
        deduped[int(row["step"])] = row
    return [deduped[step] for step in sorted(deduped)]


def scale_point(value: float, minimum: float, maximum: float, span: float) -> float:
    if maximum <= minimum:
        return span / 2.0
    return (value - minimum) / (maximum - minimum) * span


def polyline_points(values: list[float], left: float, top: float, width: float, height: float) -> str:
    x_min = 0
    x_max = max(len(values) - 1, 1)
    y_min = min(values)
    y_max = max(values)

    points: list[str] = []
    for index, value in enumerate(values):
        x = left + scale_point(float(index), x_min, x_max, width)
        y = top + height - scale_point(value, y_min, y_max, height)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def render_panel(
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    subtitle: str,
    values: list[float],
    color: str,
) -> str:
    y_min = min(values)
    y_max = max(values)
    latest = values[-1]
    chart_top = y + 70
    chart_height = height - 94

    grid_lines = []
    for idx in range(4):
        gy = chart_top + chart_height * idx / 3
        grid_lines.append(
            f'<line x1="{x + 18:.1f}" y1="{gy:.1f}" x2="{x + width - 18:.1f}" y2="{gy:.1f}" '
            'stroke="#D8DEE6" stroke-width="1" stroke-dasharray="4 6" />'
        )

    points = polyline_points(values, x + 18, chart_top, width - 36, chart_height)
    area_points = f"{x + 18:.2f},{chart_top + chart_height:.2f} {points} {x + width - 18:.2f},{chart_top + chart_height:.2f}"

    return f"""
  <g>
    <rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="18" fill="#FFFFFF" stroke="#D8DEE6" />
    <text x="{x + 20:.1f}" y="{y + 30:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="18" font-weight="700" fill="#132238">{label}</text>
    <text x="{x + 20:.1f}" y="{y + 52:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#5B6876">{subtitle}</text>
    <text x="{x + width - 20:.1f}" y="{y + 30:.1f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="18" font-weight="700" fill="{color}">{latest:.3f}</text>
    <text x="{x + width - 20:.1f}" y="{y + 52:.1f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#5B6876">min {y_min:.3f} / max {y_max:.3f}</text>
    {''.join(grid_lines)}
    <polygon points="{area_points}" fill="{color}" opacity="0.12" />
    <polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
  </g>
"""


def render_svg(rows: list[dict[str, float]], title: str) -> str:
    steps = [int(row["step"]) for row in rows]
    loss_total = [float(row["loss_total"]) for row in rows]
    step_time_ms = [1000.0 * float(row["step_time_sec"]) for row in rows]
    throughput = [float(row["frames_per_second"]) for row in rows]
    logged_steps = ", ".join(str(step) for step in steps)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="780" viewBox="0 0 1200 780" role="img" aria-labelledby="title desc">
  <title id="title">{title}</title>
  <desc id="desc">A summary of world model training metrics across {len(rows)} logged steps.</desc>
  <rect width="1200" height="780" fill="#F4F7FB" />
  <defs>
    <linearGradient id="hero" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F6CBD" />
      <stop offset="100%" stop-color="#1F9D8B" />
    </linearGradient>
  </defs>
  <rect x="48" y="32" width="1104" height="132" rx="24" fill="url(#hero)" />
  <text x="84" y="86" font-family="Helvetica, Arial, sans-serif" font-size="34" font-weight="700" fill="#FFFFFF">{title}</text>
  <text x="84" y="116" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#EAF3FB">Compact training metrics rendered from JSONL step logs.</text>
  <text x="84" y="142" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#D7E8F7">Logged steps: {logged_steps}</text>
  {render_panel(48, 188, 1104, 176, "Total Loss", "Loss across logged steps", loss_total, "#D1495B")}
  {render_panel(48, 392, 532, 320, "Step Time (ms)", "Latency per optimizer step", step_time_ms, "#0F6CBD")}
  {render_panel(620, 392, 532, 320, "Frames / Second", "Global sequence throughput", throughput, "#1F9D8B")}
</svg>
"""


def main() -> None:
    args = parse_args()
    rows = unique_by_step(read_rows(args.input))
    svg = render_svg(rows, args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
