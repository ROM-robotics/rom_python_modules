"""Reusable Spyder-like interactive DataFrame / Sub-Data Explorer module.

Importable Python module (no circular script dependencies) located in `src/rom_python_modules/`:
    from rom_python_modules.show_spyder_like_table import show_structured_data_explorer, show_spyder_table

Features:
- Maximized window on open (`-zoomed` / full-screen geometry).
- Exact Spyder Variable Explorer Pink -> Violet -> Periwinkle-Blue HSV cell coloring.
- Toggleable `Background color` and `Column min/max` (per-column vs global normalization).
- Per-column inline `Filter` boxes (`> 0.5`, `<= -0.1`, `0.1..0.8`, or text substring).
- Clickable column headers for sorting (`asc` / `desc` / reset) + draggable column borders.
- Dynamic Row <-> Column switching (`Transpose`), column subset/reorder dialog, and row slicing.
- Deep sub-data drill-down (double-click nested dicts/lists/trials/samples + Back/Root history
  + Spyder-style pop-out sub-windows) and dedicated Observation / Action Space views.
"""
from __future__ import annotations

import colorsys
import fnmatch
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "src/B_how_to_train_your_dragon/00_neural_train/data"
DEFAULT_SPLITS = ("train", "validation", "test")
FLOAT_FORMATS = ("%.6g", "%.6f", "%.4f", "%.3e")

# Color Profile 1 Rules (03_python_gui_rules.md)
CP1_BTN_BG = "#2AFEFF"
CP1_BTN_ACTIVE_BG = "#6BFEFF"
CP1_BTN_FG = "#212B44"
CP1_DISABLED_BTN_BG = "#2C3959"
CP1_BG = "#212B44"
CP1_SURFACE_BG = "#182034"
CP1_BORDER = "#364463"
CP1_LABEL_FG = "#5C6881"
CP1_VALUE_FG = "#E2E8F0"

# Backward-compatible aliases for importers
LUXURY_BLUE = CP1_BTN_BG
LUXURY_BLUE_ACTIVE = CP1_BTN_ACTIVE_BG
DISABLED_BTN_BG = CP1_DISABLED_BTN_BG

DEFAULT_OBSERVATION_SPACE_INFO: Dict[str, Tuple[str, str]] = {
    "ex_norm": ("[-1.5, 1.5]", "Horizontal image error ex / cx"),
    "ey_norm": ("[-1.5, 1.5]", "Vertical image error ey / cy"),
    "los_v_norm": ("[-2.0, 2.0]", "Vertical LOS elevation / (pi/4)"),
    "dlos_v_norm": ("[-2.0, 2.0]", "3-step causal vertical LOS rate / 2.0"),
    "dlos_z_norm": ("[-2.0, 2.0]", "3-step causal horizontal LOS rate / 2.0"),
    "w_norm": ("[-1.0, 2.0]", "Normalized bbox width (detect_w - 100) / 150"),
    "dw_norm": ("[-2.0, 2.0]", "3-step causal bbox width rate / 100"),
    "v_horiz_norm": ("[0.0, 1.5]", "Horizontal speed hypot(vx, vy) / 8.0"),
    "vz_norm": ("[-1.5, 1.5]", "Measured NED Down velocity vz / 5.0"),
    "los_azimuth_sin": ("[-1.0, 1.0]", "sin(los_angle_z) North component"),
    "los_azimuth_cos": ("[-1.0, 1.0]", "cos(los_angle_z) East component"),
    "vx_norm": ("[-1.5, 1.5]", "Measured NED North velocity vx / 8.0"),
    "vy_norm": ("[-1.5, 1.5]", "Measured NED East velocity vy / 8.0"),
}
DEFAULT_ACTION_SPACE_INFO: Dict[str, Tuple[str, str]] = {
    "action_vx_ned_mps": ("m/s (NED North)", "Commanded North velocity setpoint"),
    "action_vy_ned_mps": ("m/s (NED East)", "Commanded East velocity setpoint"),
    "action_vz_ned_mps": ("m/s (NED Down)", "Commanded Down velocity setpoint"),
    "action_yaw_rate_radps": ("rad/s (NED Down)", "Commanded yaw-rate setpoint"),
}
DEFAULT_SIDECAR_SCHEMA: Dict[str, List[str]] = {
    "runner_mode": ["runner_mode_str"],
    "px4_matched": ["is_px4_row_matched"],
    "px4_timestamp_us": ["px4_timestamp_us"],
    "px4_rpy_rad": ["roll_rad", "pitch_rad", "yaw_rad"],
    "px4_accel_ned_mps2": ["ax_ned_mps2", "ay_ned_mps2", "az_ned_mps2"],
    "px4_ang_vel_radps": ["p_body_radps", "q_body_radps", "r_body_radps"],
    "px4_traj_sp": ["vx_sp_ned_mps", "vy_sp_ned_mps", "vz_sp_ned_mps", "yawspeed_sp_radps"],
    "px4_traj_sp_valid": ["is_traj_sp_valid"],
    "stats_pos_ned_m": [
        "self_x_ned_m",
        "self_y_ned_m",
        "self_z_ned_m",
        "target_x_ned_m",
        "target_y_ned_m",
        "target_z_ned_m",
        "dist_m",
    ],
}

__all__ = [
    "FLOAT_FORMATS",
    "GridTableModel",
    "StructuredDatasetWorkspace",
    "show_spyder_table",
    "show_structured_data_explorer",
    "spyder_cell_bg_hex",
]


def spyder_cell_bg_hex(norm_val: float) -> str:
    """Return Spyder Variable Explorer HSV heatmap hex color (0=coral-red, 0.5=pink, 1=blue)."""
    if not math.isfinite(norm_val):
        return "#f3f4f6"
    clamped = max(0.0, min(1.0, norm_val))
    hue = (1.0 - 0.33 * clamped) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.42, 0.98)
    return f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"


def _format_cell_value(val: Any, float_fmt: str = "%.6g") -> str:
    if val is None:
        return ""
    if isinstance(val, (bool, np.bool_)):
        return "True" if bool(val) else "False"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        fval = float(val)
        if not math.isfinite(fval):
            return str(fval)
        try:
            return float_fmt % fval
        except (TypeError, ValueError):
            return f"{fval:.6g}"
    return str(val)


def _parse_slice_spec(spec: str, length: int) -> np.ndarray:
    """Parse 'start:stop:step' or single integer into row index array."""
    cleaned = spec.strip()
    if not cleaned:
        return np.arange(length, dtype=np.int64)
    if ":" not in cleaned:
        idx = int(cleaned)
        if idx < 0:
            idx += length
        if 0 <= idx < length:
            return np.asarray([idx], dtype=np.int64)
        return np.zeros(0, dtype=np.int64)
    parts = [p.strip() for p in cleaned.split(":")]
    if len(parts) > 3:
        raise ValueError("Slice must have at most 3 parts: start:stop:step")
    vals = [int(p) if p else None for p in parts]
    while len(vals) < 3:
        vals.append(None)
    sl = slice(vals[0], vals[1], vals[2])
    return np.arange(length, dtype=np.int64)[sl]


def _evaluate_column_filter(arr: np.ndarray, is_numeric: bool, expr: str) -> np.ndarray:
    """Evaluate Spyder-like column filter expression on a 1D array."""
    q = expr.strip()
    n = len(arr)
    if not q or n == 0:
        return np.ones(n, dtype=bool)

    if is_numeric:
        farr = arr.astype(np.float64, copy=False)
        range_match = re.match(r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\.\.\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)$", q)
        if range_match:
            lo, hi = float(range_match.group(1)), float(range_match.group(2))
            return (farr >= lo) & (farr <= hi)

        op_match = re.match(r"^(<=|>=|!=|==|<|>|=)\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)$", q)
        if op_match:
            op, rhs = op_match.group(1), float(op_match.group(2))
            if op == "<":
                return farr < rhs
            if op == "<=":
                return farr <= rhs
            if op == ">":
                return farr > rhs
            if op == ">=":
                return farr >= rhs
            if op == "!=":
                return farr != rhs
            return np.isclose(farr, rhs, rtol=1e-6, atol=1e-9)

    q_low = q.lower()
    return np.asarray([q_low in str(v).lower() for v in arr], dtype=bool)


SubDataFactory = Callable[[], "GridTableModel"]
RowSubDataFn = Callable[[int], Optional[Tuple[str, "GridTableModel"]]]


class GridTableModel:
    """2D-virtualized Spyder DataFrame model supporting filter, sort, slice, col select & transpose."""

    def __init__(
        self,
        title: str,
        col_names: Sequence[str],
        col_arrays: Sequence[np.ndarray],
        index_labels: Optional[Sequence[Any]] = None,
        subtitle: str = "",
        col_tags: Optional[Dict[str, str]] = None,
        col_descriptions: Optional[Dict[str, str]] = None,
        cell_subdata: Optional[Dict[Tuple[int, int], Tuple[str, SubDataFactory]]] = None,
        row_subdata_fn: Optional[RowSubDataFn] = None,
    ) -> None:
        if len(col_names) != len(col_arrays):
            raise ValueError("col_names and col_arrays length mismatch")
        self.title = title
        self.subtitle = subtitle
        self.all_col_names = list(col_names)
        self.col_arrays: List[np.ndarray] = [np.asarray(a) for a in col_arrays]
        self.raw_row_count = len(self.col_arrays[0]) if self.col_arrays else 0
        for arr in self.col_arrays:
            if len(arr) != self.raw_row_count:
                raise ValueError("All column arrays must share the same row count")

        self.col_is_numeric: List[bool] = [
            arr.dtype.kind in ("i", "u", "f") for arr in self.col_arrays
        ]
        if index_labels is not None:
            self.raw_index_labels = np.asarray(index_labels)
        else:
            self.raw_index_labels = np.arange(self.raw_row_count, dtype=np.int64)

        self.col_tags = dict(col_tags or {})
        self.col_descriptions = dict(col_descriptions or {})
        self.cell_subdata = dict(cell_subdata or {})
        self.row_subdata_fn = row_subdata_fn

        self.visible_col_indices: List[int] = list(range(len(self.all_col_names)))
        self.col_filters: Dict[int, str] = {}
        self.transposed_col_filters: Dict[int, str] = {}
        self.slice_spec: str = ""
        self.sort_col_raw: Optional[int] = None
        self.sort_desc: bool = False
        self.transposed: bool = False

        self._active_rows: np.ndarray = np.arange(self.raw_row_count, dtype=np.int64)
        self._active_cols: List[int] = list(self.visible_col_indices)
        self._col_minmax: Dict[int, Tuple[float, float]] = {}
        self._transposed_col_minmax: Dict[int, Tuple[float, float]] = {}
        self._global_minmax: Tuple[float, float] = (0.0, 1.0)
        self._recompute_view()

    @classmethod
    def from_dict_columns(
        cls,
        title: str,
        columns: Dict[str, Any],
        index_labels: Optional[Sequence[Any]] = None,
        subtitle: str = "",
        col_tags: Optional[Dict[str, str]] = None,
        col_descriptions: Optional[Dict[str, str]] = None,
        cell_subdata: Optional[Dict[Tuple[int, int], Tuple[str, SubDataFactory]]] = None,
        row_subdata_fn: Optional[RowSubDataFn] = None,
    ) -> "GridTableModel":
        names = list(columns.keys())
        arrays = [np.asarray(columns[k]) for k in names]
        return cls(
            title=title,
            col_names=names,
            col_arrays=arrays,
            index_labels=index_labels,
            subtitle=subtitle,
            col_tags=col_tags,
            col_descriptions=col_descriptions,
            cell_subdata=cell_subdata,
            row_subdata_fn=row_subdata_fn,
        )

    @classmethod
    def from_nested_object(
        cls,
        title: str,
        obj: Any,
        subtitle: str = "",
        trial_jump_fn: Optional[Callable[[str, str], Optional[Tuple[str, "GridTableModel"]]]] = None,
    ) -> "GridTableModel":
        """Build a recursive drillable GridTableModel from arbitrary dict / list / ndarray."""
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                return cls.from_dict_columns(
                    title=title,
                    columns={"value": obj},
                    subtitle=subtitle or f"1D ndarray shape={obj.shape} dtype={obj.dtype}",
                )
            if obj.ndim == 2:
                cols = {f"col_{j}": obj[:, j] for j in range(obj.shape[1])}
                return cls.from_dict_columns(
                    title=title,
                    columns=cols,
                    subtitle=subtitle or f"2D ndarray shape={obj.shape} dtype={obj.dtype}",
                )

        if isinstance(obj, list) and obj and all(isinstance(item, dict) for item in obj):
            keys: List[str] = []
            seen = set()
            for item in obj:
                for k in item.keys():
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)
            col_data: Dict[str, List[Any]] = {k: [] for k in keys}
            cell_sub: Dict[Tuple[int, int], Tuple[str, SubDataFactory]] = {}
            for r_idx, item in enumerate(obj):
                for c_idx, k in enumerate(keys):
                    val = item.get(k)
                    if isinstance(val, (dict, list)):
                        summary = (
                            f"▶ {{dict: {len(val)} keys}}"
                            if isinstance(val, dict)
                            else f"▶ [list: {len(val)} items]"
                        )
                        col_data[k].append(summary)
                        sub_title = f"{title}[{r_idx}].{k}"
                        cell_sub[(r_idx, c_idx)] = (
                            sub_title,
                            lambda v=val, st=sub_title: cls.from_nested_object(st, v, trial_jump_fn=trial_jump_fn),
                        )
                    else:
                        col_data[k].append(val if val is not None else np.nan)

            col_arrays: List[np.ndarray] = []
            for k in keys:
                vals = col_data[k]
                if all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in vals):
                    col_arrays.append(np.asarray(vals, dtype=np.float64))
                else:
                    col_arrays.append(np.asarray(["" if v is None else str(v) for v in vals], dtype=object))

            def _row_drill(raw_r: int) -> Optional[Tuple[str, GridTableModel]]:
                row_dict = obj[raw_r]
                if trial_jump_fn is not None and "run_id" in row_dict and "split" in row_dict:
                    jumped = trial_jump_fn(str(row_dict["split"]), str(row_dict["run_id"]))
                    if jumped is not None:
                        return jumped
                sub_t = f"{title}[{raw_r}]"
                return sub_t, cls.from_nested_object(sub_t, row_dict, trial_jump_fn=trial_jump_fn)

            return cls(
                title=title,
                col_names=keys,
                col_arrays=col_arrays,
                subtitle=subtitle or f"Table of {len(obj)} records (double-click row or ▶ cell to drill in)",
                cell_subdata=cell_sub,
                row_subdata_fn=_row_drill,
            )

        if isinstance(obj, dict):
            keys_list = list(obj.keys())
            types_list, shapes_list, previews = [], [], []
            cell_sub = {}
            for r_idx, k in enumerate(keys_list):
                v = obj[k]
                types_list.append(type(v).__name__)
                if isinstance(v, dict):
                    shapes_list.append(f"{len(v)} keys")
                    previews.append(f"▶ {{ {', '.join(list(map(str, v.keys()))[:5])} }}")
                    sub_t = f"{title}.{k}"
                    cell_sub[(r_idx, 3)] = (
                        sub_t,
                        lambda val=v, st=sub_t: cls.from_nested_object(st, val, trial_jump_fn=trial_jump_fn),
                    )
                elif isinstance(v, list):
                    shapes_list.append(f"{len(v)} items")
                    previews.append(f"▶ [list of {len(v)} items — double-click to enter]")
                    sub_t = f"{title}.{k}"
                    cell_sub[(r_idx, 3)] = (
                        sub_t,
                        lambda val=v, st=sub_t: cls.from_nested_object(st, val, trial_jump_fn=trial_jump_fn),
                    )
                elif isinstance(v, np.ndarray):
                    shapes_list.append(str(v.shape))
                    previews.append(f"▶ ndarray(shape={v.shape}, dtype={v.dtype})")
                    sub_t = f"{title}.{k}"
                    cell_sub[(r_idx, 3)] = (
                        sub_t,
                        lambda val=v, st=sub_t: cls.from_nested_object(st, val, trial_jump_fn=trial_jump_fn),
                    )
                else:
                    shapes_list.append("scalar")
                    previews.append(str(v))

            def _dict_row_drill(raw_r: int) -> Optional[Tuple[str, GridTableModel]]:
                if (raw_r, 3) in cell_sub:
                    lbl, fn = cell_sub[(raw_r, 3)]
                    return lbl, fn()
                return None

            return cls(
                title=title,
                col_names=["Key", "Type", "Size / Shape", "Value / Sub-Data (Double-Click ▶)"],
                col_arrays=[
                    np.asarray(keys_list, dtype=object),
                    np.asarray(types_list, dtype=object),
                    np.asarray(shapes_list, dtype=object),
                    np.asarray(previews, dtype=object),
                ],
                subtitle=subtitle or f"Dictionary with {len(keys_list)} entries",
                cell_subdata=cell_sub,
                row_subdata_fn=_dict_row_drill,
            )

        if isinstance(obj, list):
            types_l = [type(x).__name__ for x in obj]
            vals_l = []
            cell_sub = {}
            for r_idx, x in enumerate(obj):
                if isinstance(x, (dict, list, np.ndarray)):
                    vals_l.append(f"▶ {type(x).__name__} (double-click to enter)")
                    sub_t = f"{title}[{r_idx}]"
                    cell_sub[(r_idx, 1)] = (
                        sub_t,
                        lambda val=x, st=sub_t: cls.from_nested_object(st, val, trial_jump_fn=trial_jump_fn),
                    )
                else:
                    vals_l.append(str(x))
            return cls(
                title=title,
                col_names=["Type", "Value"],
                col_arrays=[np.asarray(types_l, dtype=object), np.asarray(vals_l, dtype=object)],
                subtitle=subtitle or f"List of {len(obj)} items",
                cell_subdata=cell_sub,
                row_subdata_fn=lambda r: (cell_sub[(r, 1)][0], cell_sub[(r, 1)][1]()) if (r, 1) in cell_sub else None,
            )

        return cls.from_dict_columns(title=title, columns={"value": [str(obj)]}, subtitle=subtitle)

    def _recompute_view(self) -> None:
        base_rows = _parse_slice_spec(self.slice_spec, self.raw_row_count)
        if len(base_rows) > 0 and self.col_filters:
            mask = np.ones(len(base_rows), dtype=bool)
            for raw_c, expr in self.col_filters.items():
                if not expr.strip():
                    continue
                sub_arr = self.col_arrays[raw_c][base_rows]
                mask &= _evaluate_column_filter(sub_arr, self.col_is_numeric[raw_c], expr)
            base_rows = base_rows[mask]

        if self.sort_col_raw is not None and len(base_rows) > 1:
            sub_arr = self.col_arrays[self.sort_col_raw][base_rows]
            if self.col_is_numeric[self.sort_col_raw]:
                order = np.argsort(sub_arr.astype(np.float64, copy=False), kind="mergesort")
            else:
                order = np.argsort(np.asarray([str(x) for x in sub_arr]), kind="mergesort")
            if self.sort_desc:
                order = order[::-1]
            base_rows = base_rows[order]

        self._active_rows = base_rows

        active_cols = list(self.visible_col_indices)
        if self.transposed and self.transposed_col_filters and len(self._active_rows) > 0:
            col_mask = np.ones(len(active_cols), dtype=bool)
            for disp_c, expr in self.transposed_col_filters.items():
                if not expr.strip() or disp_c >= len(self._active_rows):
                    continue
                raw_r = int(self._active_rows[disp_c])
                vals = np.asarray([self.col_arrays[c][raw_r] for c in active_cols], dtype=object)
                all_num = all(self.col_is_numeric[c] for c in active_cols)
                if all_num:
                    vals = vals.astype(np.float64)
                col_mask &= _evaluate_column_filter(vals, all_num, expr)
            active_cols = [c for c, keep in zip(active_cols, col_mask) if keep]
        self._active_cols = active_cols

        self._col_minmax.clear()
        self._transposed_col_minmax.clear()
        g_min, g_max = float("inf"), float("-inf")
        if len(self._active_rows) > 0:
            for c in self._active_cols:
                if not self.col_is_numeric[c]:
                    continue
                sub = self.col_arrays[c][self._active_rows].astype(np.float64, copy=False)
                finite = sub[np.isfinite(sub)]
                if len(finite) > 0:
                    cmin, cmax = float(np.min(finite)), float(np.max(finite))
                    self._col_minmax[c] = (cmin, cmax)
                    g_min = min(g_min, cmin)
                    g_max = max(g_max, cmax)
        self._global_minmax = (g_min, g_max) if g_min <= g_max else (0.0, 1.0)

    def shape(self) -> Tuple[int, int]:
        if not self.transposed:
            return len(self._active_rows), len(self._active_cols)
        return len(self._active_cols), len(self._active_rows)

    def toggle_transpose(self) -> bool:
        self.transposed = not self.transposed
        self._recompute_view()
        return self.transposed

    def set_slice_spec(self, spec: str) -> None:
        _parse_slice_spec(spec, self.raw_row_count)
        self.slice_spec = spec.strip()
        self._recompute_view()

    def set_visible_columns(self, col_indices: Sequence[int]) -> None:
        if not col_indices:
            raise ValueError("Select at least one column.")
        self.visible_col_indices = [int(i) for i in col_indices]
        self._recompute_view()

    def set_filter(self, disp_c: int, expr: str) -> None:
        if not self.transposed:
            if 0 <= disp_c < len(self._active_cols):
                raw_c = self._active_cols[disp_c]
                if expr.strip():
                    self.col_filters[raw_c] = expr.strip()
                else:
                    self.col_filters.pop(raw_c, None)
        else:
            if expr.strip():
                self.transposed_col_filters[disp_c] = expr.strip()
            else:
                self.transposed_col_filters.pop(disp_c, None)
        self._recompute_view()

    def get_filter(self, disp_c: int) -> str:
        if not self.transposed:
            if 0 <= disp_c < len(self._active_cols):
                return self.col_filters.get(self._active_cols[disp_c], "")
            return ""
        return self.transposed_col_filters.get(disp_c, "")

    def clear_filters_and_slice(self) -> None:
        self.col_filters.clear()
        self.transposed_col_filters.clear()
        self.slice_spec = ""
        self.sort_col_raw = None
        self.sort_desc = False
        self._recompute_view()

    def toggle_sort_column(self, disp_c: int) -> None:
        if self.transposed:
            return
        if not (0 <= disp_c < len(self._active_cols)):
            return
        raw_c = self._active_cols[disp_c]
        if self.sort_col_raw != raw_c:
            self.sort_col_raw = raw_c
            self.sort_desc = False
        elif not self.sort_desc:
            self.sort_desc = True
        else:
            self.sort_col_raw = None
            self.sort_desc = False
        self._recompute_view()

    def disp_col_header(self, disp_c: int) -> str:
        if not self.transposed:
            raw_c = self._active_cols[disp_c]
            name = self.all_col_names[raw_c]
            tag = self.col_tags.get(name, "")
            prefix = f"[{tag}] " if tag else ""
            arrow = ""
            if self.sort_col_raw == raw_c:
                arrow = " ▼" if self.sort_desc else " ▲"
            return f"{prefix}{name}{arrow}"
        raw_r = int(self._active_rows[disp_c])
        return f"#{self.raw_index_labels[raw_r]}"

    def disp_row_label(self, disp_r: int) -> str:
        if not self.transposed:
            raw_r = int(self._active_rows[disp_r])
            return str(self.raw_index_labels[raw_r])
        raw_c = self._active_cols[disp_r]
        name = self.all_col_names[raw_c]
        tag = self.col_tags.get(name, "")
        return f"[{tag}] {name}" if tag else name

    def _map_coords(self, disp_r: int, disp_c: int) -> Tuple[int, int]:
        if not self.transposed:
            return int(self._active_rows[disp_r]), self._active_cols[disp_c]
        return int(self._active_rows[disp_c]), self._active_cols[disp_r]

    def cell_info(self, disp_r: int, disp_c: int) -> Tuple[Any, Optional[float], bool]:
        raw_r, raw_c = self._map_coords(disp_r, disp_c)
        val = self.col_arrays[raw_c][raw_r]
        num_val: Optional[float] = None
        if self.col_is_numeric[raw_c]:
            fval = float(val)
            if math.isfinite(fval):
                num_val = fval
        has_sub = (raw_r, raw_c) in self.cell_subdata or self.row_subdata_fn is not None
        return val, num_val, has_sub

    def _transposed_col_bounds(self, disp_c: int) -> Tuple[float, float]:
        if disp_c in self._transposed_col_minmax:
            return self._transposed_col_minmax[disp_c]
        raw_r = int(self._active_rows[disp_c])
        nums = [
            float(self.col_arrays[c][raw_r])
            for c in self._active_cols
            if self.col_is_numeric[c] and math.isfinite(float(self.col_arrays[c][raw_r]))
        ]
        bounds = (min(nums), max(nums)) if nums else (0.0, 1.0)
        self._transposed_col_minmax[disp_c] = bounds
        return bounds

    def cell_bg_color(self, disp_r: int, disp_c: int, bg_enabled: bool, col_minmax: bool) -> str:
        if not bg_enabled:
            return "#ffffff" if (disp_r % 2 == 0) else "#f8fafc"
        _, num_val, _ = self.cell_info(disp_r, disp_c)
        if num_val is None:
            return "#ffffff" if (disp_r % 2 == 0) else "#f8fafc"

        if col_minmax:
            if not self.transposed:
                raw_c = self._active_cols[disp_c]
                lo, hi = self._col_minmax.get(raw_c, (0.0, 1.0))
            else:
                lo, hi = self._transposed_col_bounds(disp_c)
        else:
            lo, hi = self._global_minmax

        norm = 0.5 if abs(hi - lo) <= 1e-12 else (num_val - lo) / (hi - lo)
        return spyder_cell_bg_hex(norm)

    def column_detail_summary(self, disp_r: int, disp_c: int) -> str:
        raw_r, raw_c = self._map_coords(disp_r, disp_c)
        col_name = self.all_col_names[raw_c]
        val = self.col_arrays[raw_c][raw_r]
        desc = self.col_descriptions.get(col_name, "")
        bounds_txt = ""
        if raw_c in self._col_minmax:
            cmin, cmax = self._col_minmax[raw_c]
            bounds_txt = f" | col min={cmin:.6g}, max={cmax:.6g}"
        desc_txt = f" ({desc})" if desc else ""
        return f"Row #{self.raw_index_labels[raw_r]}, Col '{col_name}'{desc_txt} = {val}{bounds_txt}"

    def drill_down(self, disp_r: int, disp_c: int) -> Optional[Tuple[str, "GridTableModel"]]:
        n_r, n_c = self.shape()
        if not (0 <= disp_r < n_r and 0 <= disp_c < n_c):
            return None
        raw_r, raw_c = self._map_coords(disp_r, disp_c)
        if (raw_r, raw_c) in self.cell_subdata:
            label, factory = self.cell_subdata[(raw_r, raw_c)]
            return label, factory()
        if self.row_subdata_fn is not None:
            return self.row_subdata_fn(raw_r)
        return None


class StructuredDatasetWorkspace:
    """Loads Two-Tier NPZ splits + meta.json into navigable Observation/Action/Sub-Data models."""

    def __init__(
        self,
        data_dir: Path,
        output_stem: Optional[str] = None,
        observation_space_info: Optional[Dict[str, Tuple[str, str]]] = None,
        action_space_info: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        if output_stem and (self.data_dir / f"{output_stem}.meta.json").is_file():
            self.meta_path = self.data_dir / f"{output_stem}.meta.json"
        else:
            candidates = sorted(self.data_dir.glob("*.meta.json"))
            if not candidates:
                raise FileNotFoundError(
                    f"Dataset metadata (*.meta.json) not found in {self.data_dir}.\n"
                    "Run extraction first to build Two-Tier NPZ files."
                )
            self.meta_path = candidates[0]

        self.stem = self.meta_path.name[:-len(".meta.json")]
        self.meta: Dict[str, Any] = json.loads(self.meta_path.read_text(encoding="utf-8"))

        declared_splits = list(self.meta.get("splits", {}).keys()) or list(DEFAULT_SPLITS)
        self.splits_order: Tuple[str, ...] = tuple(declared_splits)
        self.splits_raw: Dict[str, Dict[str, np.ndarray]] = {}
        for split in self.splits_order:
            npz_path = self.data_dir / f"{self.stem}.{split}.npz"
            if npz_path.is_file():
                with np.load(npz_path, allow_pickle=False) as d:
                    self.splits_raw[split] = {k: d[k].copy() for k in d.files}
        if not self.splits_raw:
            raise FileNotFoundError(f"No {self.stem}.*.npz split files found in {self.data_dir}")

        first_split = next(iter(self.splits_raw.values()))
        self.feature_columns: List[str] = list(
            self.meta.get("feature_names")
            or (first_split["feature_names"].tolist() if "feature_names" in first_split else list(DEFAULT_OBSERVATION_SPACE_INFO.keys()))
        )
        self.action_columns: List[str] = list(
            self.meta.get("label_names")
            or (first_split["label_names"].tolist() if "label_names" in first_split else list(DEFAULT_ACTION_SPACE_INFO.keys()))
        )
        self.sidecar_schema: Dict[str, List[str]] = dict(
            self.meta.get("sidecar_arrays") or DEFAULT_SIDECAR_SCHEMA
        )
        self.obs_info: Dict[str, Tuple[str, str]] = {
            **DEFAULT_OBSERVATION_SPACE_INFO,
            **(observation_space_info or {}),
        }
        self.act_info: Dict[str, Tuple[str, str]] = {
            **DEFAULT_ACTION_SPACE_INFO,
            **(action_space_info or {}),
        }

        self.col_tags: Dict[str, str] = {}
        self.col_descriptions: Dict[str, str] = {}
        for f_name in self.feature_columns:
            self.col_tags[f_name] = "OBS"
            bounds, desc = self.obs_info.get(f_name, ("normalized", "Observation feature"))
            self.col_descriptions[f_name] = f"Observation Space {bounds}: {desc}"
        for a_name in self.action_columns:
            self.col_tags[a_name] = "ACT"
            unit, desc = self.act_info.get(a_name, ("physical", "Action label"))
            self.col_descriptions[a_name] = f"Action Space [{unit}]: {desc}"
        for group_key, sub_cols in self.sidecar_schema.items():
            for sc in sub_cols:
                self.col_tags[sc] = "PX4" if group_key.startswith("px4_") else "STAT"
                self.col_descriptions[sc] = f"Tier 2 Sidecar ({group_key})"

    def default_split(self) -> str:
        for s in self.splits_order:
            if s in self.splits_raw and len(self.splits_raw[s]["X"]) > 0:
                return s
        return next(iter(self.splits_raw.keys()))

    def build_space_spec_table(self) -> GridTableModel:
        """Build Feature (Observation Space) + Label (Action Space) contract & split stats table."""
        names = list(self.feature_columns) + list(self.action_columns)
        spaces = ["Observation (Feature)"] * len(self.feature_columns) + ["Action (Label)"] * len(self.action_columns)
        indices = [f"X[:, {i}]" for i in range(len(self.feature_columns))] + [
            f"Y[:, {i}]" for i in range(len(self.action_columns))
        ]
        units, meanings = [], []
        for n in self.feature_columns:
            u, m = self.obs_info.get(n, ("normalized", "Observation feature"))
            units.append(u)
            meanings.append(m)
        for n in self.action_columns:
            u, m = self.act_info.get(n, ("physical", "Action label"))
            units.append(u)
            meanings.append(m)

        cols: Dict[str, Any] = {
            "space": np.asarray(spaces, dtype=object),
            "tensor_slot": np.asarray(indices, dtype=object),
            "name": np.asarray(names, dtype=object),
            "bounds_or_unit": np.asarray(units, dtype=object),
        }
        for split in self.splits_order:
            if split not in self.splits_raw or len(self.splits_raw[split]["X"]) == 0:
                continue
            xy = np.hstack([self.splits_raw[split]["X"], self.splits_raw[split]["Y"]]).astype(np.float64)
            cols[f"{split}_mean"] = np.mean(xy, axis=0)
            cols[f"{split}_std"] = np.std(xy, axis=0)
            cols[f"{split}_min"] = np.min(xy, axis=0)
            cols[f"{split}_max"] = np.max(xy, axis=0)
        cols["description"] = np.asarray(meanings, dtype=object)
        return GridTableModel.from_dict_columns(
            title=f"Observation Space ({len(self.feature_columns)} Features) & Action Space ({len(self.action_columns)} Labels) Specification",
            columns=cols,
            index_labels=names,
            subtitle="Complete Observation Space (X) and Physical Action Space (Y) summary across splits",
        )

    def build_split_stats_table(self, split: str) -> GridTableModel:
        d = self.splits_raw[split]
        xy = np.hstack([d["X"], d["Y"]]).astype(np.float64)
        names = list(self.feature_columns) + list(self.action_columns)
        spaces = ["Observation"] * len(self.feature_columns) + ["Action"] * len(self.action_columns)
        if len(xy) == 0:
            zeros = np.zeros(len(names), dtype=np.float64)
            stats_cols = {"space": spaces, "min": zeros, "p5": zeros, "mean": zeros, "std": zeros, "p50": zeros, "p95": zeros, "max": zeros}
        else:
            stats_cols = {
                "space": np.asarray(spaces, dtype=object),
                "min": np.min(xy, axis=0),
                "p5": np.percentile(xy, 5, axis=0),
                "mean": np.mean(xy, axis=0),
                "std": np.std(xy, axis=0),
                "p50": np.percentile(xy, 50, axis=0),
                "p95": np.percentile(xy, 95, axis=0),
                "max": np.max(xy, axis=0),
            }
        return GridTableModel.from_dict_columns(
            title=f"[{split}] Space Distribution Statistics ({len(names)} dims)",
            columns=stats_cols,
            index_labels=names,
            subtitle=f"{split} split: {len(xy):,} samples across {len(set(d['run_id'].tolist()))} trials",
        )

    def _expand_sidecar_columns(self, d: Dict[str, np.ndarray], mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for key in ("px4_matched", "px4_timestamp_us", "px4_rpy_rad", "px4_accel_ned_mps2", "px4_ang_vel_radps", "px4_traj_sp", "px4_traj_sp_valid", "stats_pos_ned_m"):
            if key not in d:
                continue
            arr = d[key][mask] if mask is not None else d[key]
            sub_names = self.sidecar_schema.get(key, [key])
            if arr.ndim == 1:
                out[sub_names[0]] = arr
            elif arr.ndim == 2:
                for j, col_name in enumerate(sub_names):
                    out[col_name] = arr[:, j]
        return out

    def build_sample_subdata_model(self, split: str, sample_idx: int) -> GridTableModel:
        """Drill-down inspector for a single sample row showing nested Observation, Action, and PX4 sub-tables."""
        d = self.splits_raw[split]
        run_id = str(d["run_id"][sample_idx])
        mode = str(d["runner_mode"][sample_idx]) if "runner_mode" in d else "unknown"
        seg_id = int(d["segment_id"][sample_idx])
        src_row = int(d["source_row"][sample_idx])
        t_s = float(d["time_s"][sample_idx])
        x_vec = d["X"][sample_idx]
        y_vec = d["Y"][sample_idx]
        feat_cols = self.feature_columns
        act_cols = self.action_columns

        def _obs_detail() -> GridTableModel:
            return GridTableModel.from_dict_columns(
                title=f"Sample #{sample_idx} — Observation Space (X [{len(feat_cols)}])",
                columns={
                    "feature_index": np.arange(len(feat_cols), dtype=np.int64),
                    "value": x_vec.astype(np.float64),
                    "clamp_bounds": np.asarray([self.obs_info.get(k, ("", ""))[0] for k in feat_cols], dtype=object),
                    "meaning": np.asarray([self.obs_info.get(k, ("", ""))[1] for k in feat_cols], dtype=object),
                },
                index_labels=feat_cols,
                subtitle=f"{split} sample #{sample_idx} | trial={run_id} ({mode}) | t={t_s:.4f}s",
            )

        def _act_detail() -> GridTableModel:
            sp_arr = d["px4_traj_sp"][sample_idx].astype(np.float64) if "px4_traj_sp" in d else np.zeros(len(act_cols))
            return GridTableModel.from_dict_columns(
                title=f"Sample #{sample_idx} — Action Space (Y [{len(act_cols)}]) vs PX4 Setpoint",
                columns={
                    "action_index": np.arange(len(act_cols), dtype=np.int64),
                    "logged_action_Y": y_vec.astype(np.float64),
                    "px4_traj_sp": sp_arr,
                    "diff": y_vec.astype(np.float64) - sp_arr,
                    "unit": np.asarray([self.act_info.get(k, ("", ""))[0] for k in act_cols], dtype=object),
                },
                index_labels=act_cols,
                subtitle=f"{split} sample #{sample_idx} | Requested Action vs PX4 TrajectorySetpoint",
            )

        def _px4_detail() -> GridTableModel:
            side = self._expand_sidecar_columns(d, np.asarray([sample_idx], dtype=np.int64))
            names = list(side.keys())
            vals = np.asarray([float(side[k][0]) for k in names], dtype=np.float64)
            return GridTableModel.from_dict_columns(
                title=f"Sample #{sample_idx} — Tier 2 PX4 Flight Log & NED Stats",
                columns={"value": vals},
                index_labels=names,
                subtitle=f"px4_matched={int(d['px4_matched'][sample_idx])}, px4_timestamp_us={int(d['px4_timestamp_us'][sample_idx])}",
            )

        obs_preview = ", ".join(f"{feat_cols[i]}={x_vec[i]:.3f}" for i in range(min(4, len(feat_cols))))
        act_preview = ", ".join(f"{act_cols[i]}={y_vec[i]:.3f}" for i in range(min(4, len(act_cols))))
        rows_meta = [
            ("run_id (Parent Trial)", "Trial Trajectory", run_id, f"▶ Double-click to inspect all samples in {run_id}"),
            ("segment_id (Parent Segment)", "Segment Trajectory", f"seg #{seg_id}", f"▶ Double-click to inspect {run_id} segment #{seg_id}"),
            ("Observation Space (X)", f"float32[{len(feat_cols)}]", f"({len(feat_cols)},)", f"▶ {obs_preview} ..."),
            ("Action Space (Y)", f"float32[{len(act_cols)}]", f"({len(act_cols)},)", f"▶ {act_preview}"),
            ("Tier 2 PX4 Sidecars", "Sidecar Group", "(20,)", f"▶ px4_matched={int(d['px4_matched'][sample_idx])}, rpy/accel/gyro/sp/pos"),
            ("runner_mode", "str", "scalar", mode),
            ("source_row", "int64", "scalar", str(src_row)),
            ("time_s", "float64", "scalar", f"{t_s:.4f}"),
            ("elapsed_s", "float64", "scalar", f"{float(d['elapsed_s'][sample_idx]):.4f}"),
        ]
        cell_sub: Dict[Tuple[int, int], Tuple[str, SubDataFactory]] = {
            (0, 3): (f"{split}:{run_id}", lambda: self.build_split_table(split, mode="full", run_id_filter=run_id)),
            (1, 3): (f"{split}:{run_id}:seg{seg_id}", lambda: self.build_split_table(split, mode="full", run_id_filter=run_id, seg_filter=seg_id)),
            (2, 3): (f"Sample #{sample_idx} Obs", _obs_detail),
            (3, 3): (f"Sample #{sample_idx} Act", _act_detail),
            (4, 3): (f"Sample #{sample_idx} PX4", _px4_detail),
        }

        def _drill_row(r: int) -> Optional[Tuple[str, GridTableModel]]:
            if (r, 3) in cell_sub:
                lbl, fn = cell_sub[(r, 3)]
                return lbl, fn()
            return None

        return GridTableModel(
            title=f"[{split}] Sample #{sample_idx} Sub-Data Breakdown",
            col_names=["Field / Space", "Type", "Shape", "Value / Sub-Data (Double-Click ▶)"],
            col_arrays=[
                np.asarray([r[0] for r in rows_meta], dtype=object),
                np.asarray([r[1] for r in rows_meta], dtype=object),
                np.asarray([r[2] for r in rows_meta], dtype=object),
                np.asarray([r[3] for r in rows_meta], dtype=object),
            ],
            subtitle=f"Trial {run_id} ({mode}), CSV line {src_row}, time_s={t_s:.4f}s — double-click any ▶ row to enter",
            cell_subdata=cell_sub,
            row_subdata_fn=_drill_row,
        )

    def build_split_table(
        self,
        split: str,
        mode: str = "obs_act",
        sidecar_key: Optional[str] = None,
        runner_mode_filter: Optional[str] = None,
        run_id_filter: Optional[str] = None,
        seg_filter: Optional[int] = None,
    ) -> GridTableModel:
        """Build virtualized Spyder table for a split (or filtered trial/mode/segment sub-data)."""
        d = self.splits_raw[split]
        n_total = len(d["X"])
        mask = np.ones(n_total, dtype=bool)
        scope_parts = [split]
        if runner_mode_filter is not None and "runner_mode" in d:
            mask &= d["runner_mode"] == runner_mode_filter
            scope_parts.append(f"mode={runner_mode_filter}")
        if run_id_filter is not None:
            mask &= d["run_id"] == run_id_filter
            scope_parts.append(f"trial={run_id_filter}")
        if seg_filter is not None:
            mask &= d["segment_id"] == seg_filter
            scope_parts.append(f"seg={seg_filter}")

        global_indices = np.nonzero(mask)[0]
        x_arr = d["X"][mask]
        y_arr = d["Y"][mask]

        cols: Dict[str, np.ndarray] = {}
        if mode == "obs":
            title_label = "Observation Space — Features (X)"
            for j, name in enumerate(self.feature_columns):
                cols[name] = x_arr[:, j]
        elif mode == "act":
            title_label = "Action Space — Labels (Y)"
            for j, name in enumerate(self.action_columns):
                cols[name] = y_arr[:, j]
        elif mode == "obs_act":
            title_label = f"Observation ({len(self.feature_columns)}) + Action ({len(self.action_columns)}) Space"
            for j, name in enumerate(self.feature_columns):
                cols[name] = x_arr[:, j]
            for j, name in enumerate(self.action_columns):
                cols[name] = y_arr[:, j]
        elif mode == "sidecar":
            if sidecar_key and sidecar_key in d:
                title_label = f"Tier 2 Sidecar — {sidecar_key}"
                arr = d[sidecar_key][mask]
                names = self.sidecar_schema.get(sidecar_key, [sidecar_key])
                if arr.ndim == 1:
                    cols[names[0]] = arr
                else:
                    for j, nm in enumerate(names):
                        cols[nm] = arr[:, j]
            else:
                title_label = "Tier 2 PX4 Flight Log & NED Stats Sidecars"
                cols.update(self._expand_sidecar_columns(d, mask))
        else:
            title_label = "Full Two-Tier Sample Table (Context + Obs + Act + PX4)"
            cols["run_id"] = d["run_id"][mask].astype(object)
            if "runner_mode" in d:
                cols["runner_mode"] = d["runner_mode"][mask].astype(object)
            cols["segment_id"] = d["segment_id"][mask]
            cols["source_row"] = d["source_row"][mask]
            cols["time_s"] = d["time_s"][mask]
            cols["elapsed_s"] = d["elapsed_s"][mask]
            for j, name in enumerate(self.feature_columns):
                cols[name] = x_arr[:, j]
            for j, name in enumerate(self.action_columns):
                cols[name] = y_arr[:, j]
            cols.update(self._expand_sidecar_columns(d, mask))

        def _row_drill(local_r: int) -> Optional[Tuple[str, GridTableModel]]:
            orig_idx = int(global_indices[local_r])
            return f"{split}[#{orig_idx}]", self.build_sample_subdata_model(split, orig_idx)

        scope_str = " / ".join(scope_parts)
        return GridTableModel.from_dict_columns(
            title=f"[{scope_str}] {title_label}",
            columns=cols,
            index_labels=global_indices,
            subtitle=(
                f"{len(global_indices):,} rows × {len(cols)} cols — "
                "Click header to sort, type in 'Filter' row to filter, 'Transpose' to swap Rows↔Cols, double-click any row for Sub-Data"
            ),
            col_tags=self.col_tags,
            col_descriptions=self.col_descriptions,
            row_subdata_fn=_row_drill,
        )

    def build_meta_root_table(self) -> GridTableModel:
        def _jump_trial(split: str, run_id: str) -> Optional[Tuple[str, GridTableModel]]:
            if split in self.splits_raw:
                mask = self.splits_raw[split]["run_id"] == run_id
                if np.any(mask):
                    return f"{split}:{run_id}", self.build_split_table(split, mode="full", run_id_filter=run_id)
            return None

        return GridTableModel.from_nested_object(
            title=self.meta_path.name,
            obj=self.meta,
            subtitle="Dataset Metadata & Embedded Sources Inventory — double-click any ▶ row to inspect nested sub-data",
            trial_jump_fn=_jump_trial,
        )

    def build_sources_table(self) -> GridTableModel:
        def _jump_trial(split: str, run_id: str) -> Optional[Tuple[str, GridTableModel]]:
            if split in self.splits_raw:
                mask = self.splits_raw[split]["run_id"] == run_id
                if np.any(mask):
                    return f"{split}:{run_id}", self.build_split_table(split, mode="full", run_id_filter=run_id)
            return None

        sources = self.meta.get("sources", [])
        return GridTableModel.from_nested_object(
            title="Trials Inventory (meta.json -> sources)",
            obj=sources,
            subtitle=(
                f"{len(sources)} source trials — double-click 'rejection_counts' to inspect rejections, "
                "or double-click any trial row to enter its samples"
            ),
            trial_jump_fn=_jump_trial,
        )


def ensure_repo_root_cwd(
    repo_root: Path = REPO_ROOT,
    script_path: Optional[Path] = None,
) -> None:
    """Enforce 03_python_gui_rules.md: Python GUI scripts must run from repo root cherry_thin."""
    cwd = Path.cwd().resolve()
    expected = repo_root.resolve()
    if cwd != expected:
        target = (script_path or Path(sys.argv[0] or __file__)).resolve()
        try:
            rel_script = target.relative_to(expected)
        except ValueError:
            rel_script = target
        raise SystemExit(
            f"[GUI Rule Error] Python GUI scripts must be run from repo root '{expected.name}':\n"
            f"  Current directory : {cwd}\n"
            f"  Expected root     : {expected}\n"
            f"  Run command       : cd {expected} && python3 {rel_script}"
        )


def apply_consistent_titlebar(win: Any) -> None:
    """Apply consistent dark titlebar decoration and #212B44 window background."""
    try:
        win.configure(background=CP1_BG)
    except Exception:  # noqa: BLE001
        pass
    win.update_idletasks()
    if sys.platform.startswith("linux"):
        try:
            wid = hex(int(win.winfo_id()))
            subprocess.run(
                [
                    "xprop",
                    "-id",
                    wid,
                    "-f",
                    "_GTK_THEME_VARIANT",
                    "8u",
                    "-set",
                    "_GTK_THEME_VARIANT",
                    "dark",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:  # noqa: BLE001
            pass


def apply_color_profile_1_style(root: Any) -> None:
    """Configure ttk.Style strictly to Color Profile 1 (#2AFEFF button, #212B44 bg, #5C6881 label)."""
    from tkinter import ttk

    apply_consistent_titlebar(root)
    root.option_add("*TCombobox*Listbox*Background", CP1_SURFACE_BG)
    root.option_add("*TCombobox*Listbox*Foreground", CP1_VALUE_FG)
    root.option_add("*TCombobox*Listbox*selectBackground", CP1_BTN_BG)
    root.option_add("*TCombobox*Listbox*selectForeground", CP1_BTN_FG)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:  # noqa: BLE001
        pass

    style.configure(".", background=CP1_BG, foreground=CP1_LABEL_FG, font=("Ubuntu", 11))
    style.configure("TFrame", background=CP1_BG)
    style.configure("TPanedwindow", background=CP1_BG)
    style.configure(
        "TLabelframe",
        background=CP1_BG,
        bordercolor=CP1_BORDER,
        lightcolor=CP1_BORDER,
        darkcolor=CP1_BORDER,
    )
    style.configure(
        "TLabelframe.Label",
        background=CP1_BG,
        foreground=CP1_LABEL_FG,
        font=("Ubuntu", 11, "bold"),
    )
    style.configure("TLabel", background=CP1_BG, foreground=CP1_LABEL_FG, font=("Ubuntu", 11))
    style.configure(
        "TCheckbutton",
        background=CP1_BG,
        foreground=CP1_LABEL_FG,
        indicatorbackground=CP1_SURFACE_BG,
        indicatorforeground=CP1_BTN_BG,
        font=("Ubuntu", 10, "bold"),
    )
    style.map(
        "TCheckbutton",
        background=[("active", CP1_BG)],
        foreground=[("active", CP1_LABEL_FG)],
    )
    style.configure(
        "TEntry",
        fieldbackground=CP1_SURFACE_BG,
        foreground=CP1_VALUE_FG,
        insertcolor=CP1_BTN_BG,
        bordercolor=CP1_BORDER,
        lightcolor=CP1_BORDER,
        darkcolor=CP1_BORDER,
        padding=(8, 6),
    )
    style.configure(
        "TCombobox",
        fieldbackground=CP1_SURFACE_BG,
        background=CP1_BG,
        foreground=CP1_VALUE_FG,
        arrowcolor=CP1_BTN_BG,
        bordercolor=CP1_BORDER,
        padding=(8, 6),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", CP1_SURFACE_BG)],
        foreground=[("readonly", CP1_VALUE_FG)],
    )
    style.configure(
        "Treeview",
        background=CP1_SURFACE_BG,
        fieldbackground=CP1_SURFACE_BG,
        foreground=CP1_VALUE_FG,
        bordercolor=CP1_BORDER,
        rowheight=32,
        font=("Ubuntu", 10),
    )
    style.map(
        "Treeview",
        background=[("selected", CP1_BTN_BG)],
        foreground=[("selected", CP1_BTN_FG)],
    )
    style.configure(
        "Treeview.Heading",
        background=CP1_BG,
        foreground=CP1_LABEL_FG,
        bordercolor=CP1_BORDER,
        font=("Ubuntu", 10, "bold"),
        padding=(8, 6),
    )
    style.map("Treeview.Heading", background=[("active", CP1_SURFACE_BG)])
    for sb_orient in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
        style.configure(
            sb_orient,
            background=CP1_BORDER,
            troughcolor=CP1_SURFACE_BG,
            bordercolor=CP1_BG,
            arrowcolor=CP1_BTN_BG,
        )


def _maximize_tk_window(win: Any) -> None:
    """Maximize a Tk or Toplevel window across Linux X11/Wayland, Windows, and macOS."""
    win.resizable(True, True)
    apply_color_profile_1_style(win)
    sw = max(1024, int(win.winfo_screenwidth()))
    sh = max(700, int(win.winfo_screenheight()))
    win.geometry(f"{sw}x{sh}+0+0")
    for attempt in (
        lambda: win.attributes("-zoomed", True),
        lambda: win.state("zoomed"),
    ):
        try:
            attempt()
            break
        except Exception:  # noqa: BLE001
            continue


def open_centered_half_dialog(parent: Any, title: str) -> Any:
    """Create a modal Toplevel sized to half of the maximized parent window and centered."""
    import tkinter as tk

    top_parent = parent.winfo_toplevel() if hasattr(parent, "winfo_toplevel") else parent
    top_parent.update_idletasks()
    pw = int(top_parent.winfo_width())
    ph = int(top_parent.winfo_height())
    if pw < 400 or ph < 300:
        pw = int(top_parent.winfo_screenwidth())
        ph = int(top_parent.winfo_screenheight())
        px, py = 0, 0
    else:
        px = int(top_parent.winfo_rootx())
        py = int(top_parent.winfo_rooty())

    dw = max(520, pw // 2)
    dh = max(360, ph // 2)
    dx = px + (pw - dw) // 2
    dy = py + (parent_h_val := ph - dh) // 2
    _ = parent_h_val

    dlg = tk.Toplevel(top_parent)
    dlg.title(title)
    dlg.transient(top_parent)
    dlg.geometry(f"{dw}x{dh}+{dx}+{dy}")
    dlg.resizable(True, True)
    apply_consistent_titlebar(dlg)
    dlg.grab_set()
    dlg.focus_set()
    return dlg


def make_luxury_button(
    parent: Any,
    text: str,
    command: Optional[Callable[[], None]] = None,
    enabled: bool = True,
    padx: int = 18,
    pady: int = 9,
    font_size: int = 10,
    min_width: int = 10,
) -> Any:
    """Create a well-sized Color Profile 1 (#2AFEFF) button with balanced 4-sided padding."""
    import tkinter as tk

    kwargs: Dict[str, Any] = {
        "text": text,
        "width": max(min_width, len(text) + 2),
        "state": "normal" if enabled else "disabled",
        "background": CP1_BTN_BG if enabled else CP1_DISABLED_BTN_BG,
        "foreground": CP1_BTN_FG,
        "activebackground": CP1_BTN_ACTIVE_BG,
        "activeforeground": CP1_BTN_FG,
        "disabledforeground": CP1_LABEL_FG,
        "padx": padx,
        "pady": pady,
        "anchor": "center",
        "justify": "center",
        "relief": "flat",
        "borderwidth": 0,
        "highlightthickness": 0,
        "font": ("Ubuntu", font_size, "bold"),
        "cursor": "hand2" if enabled else "arrow",
    }
    if command is not None:
        kwargs["command"] = command
    return tk.Button(parent, **kwargs)


def set_luxury_button_state(btn: Any, enabled: bool) -> None:
    """Toggle enabled/disabled state on a Color Profile 1 button while preserving colors."""
    btn.configure(
        state="normal" if enabled else "disabled",
        background=CP1_BTN_BG if enabled else CP1_DISABLED_BTN_BG,
        cursor="hand2" if enabled else "arrow",
    )


def show_half_message_dialog(
    parent: Any,
    title: str,
    message: str,
    is_error: bool = False,
) -> None:
    """Half-size centered modal message dialog replacing native tkinter.messagebox."""
    import tkinter as tk
    from tkinter import ttk

    dlg = open_centered_half_dialog(parent, title)
    body = ttk.Frame(dlg, padding=18)
    body.grid(row=0, column=0, sticky="nsew")
    dlg.rowconfigure(0, weight=1)
    dlg.columnconfigure(0, weight=1)
    body.columnconfigure(0, weight=1)
    body.rowconfigure(1, weight=1)

    ttk.Label(
        body,
        text=title,
        font=("Ubuntu", 12, "bold"),
        foreground=CP1_LABEL_FG,
    ).grid(row=0, column=0, sticky="w", pady=(0, 8))

    txt = tk.Text(
        body,
        wrap="word",
        font=("Ubuntu Mono", 10),
        background=CP1_SURFACE_BG,
        foreground=CP1_VALUE_FG,
        highlightthickness=0,
        borderwidth=1,
        relief="solid",
        padx=10,
        pady=10,
    )
    txt.grid(row=1, column=0, sticky="nsew")
    txt.insert("1.0", message)
    txt.configure(state="disabled")
    vsb = ttk.Scrollbar(body, orient="vertical", command=txt.yview)
    vsb.grid(row=1, column=1, sticky="ns")
    txt.configure(yscrollcommand=vsb.set)

    btn_bar = ttk.Frame(body)
    btn_bar.grid(row=2, column=0, columnspan=2, pady=(14, 4))
    make_luxury_button(btn_bar, "OK", dlg.destroy, padx=28, pady=10, font_size=10, min_width=14).grid(
        row=0, column=0, padx=8, pady=4
    )
    dlg.bind("<Return>", lambda _e: dlg.destroy())
    dlg.bind("<Escape>", lambda _e: dlg.destroy())
    dlg.wait_window()


def ask_yes_no_half_dialog(parent: Any, title: str, message: str) -> bool:
    """Half-size centered Yes/No confirmation dialog replacing messagebox.askyesno."""
    import tkinter as tk
    from tkinter import ttk

    dlg = open_centered_half_dialog(parent, title)
    answer = {"yes": False}

    body = ttk.Frame(dlg, padding=18)
    body.grid(row=0, column=0, sticky="nsew")
    dlg.rowconfigure(0, weight=1)
    dlg.columnconfigure(0, weight=1)
    body.columnconfigure(0, weight=1)
    body.rowconfigure(1, weight=1)

    ttk.Label(body, text=title, font=("Ubuntu", 12, "bold"), foreground=CP1_LABEL_FG).grid(
        row=0, column=0, sticky="w", pady=(0, 8)
    )
    txt = tk.Text(
        body,
        wrap="word",
        font=("Ubuntu", 11),
        background=CP1_SURFACE_BG,
        foreground=CP1_VALUE_FG,
        highlightthickness=0,
        borderwidth=1,
        relief="solid",
        padx=10,
        pady=10,
    )
    txt.grid(row=1, column=0, sticky="nsew")
    txt.insert("1.0", message)
    txt.configure(state="disabled")

    def choose_yes() -> None:
        answer["yes"] = True
        dlg.destroy()

    btn_bar = ttk.Frame(body)
    btn_bar.grid(row=2, column=0, pady=(14, 4))
    btn_bar.columnconfigure(0, weight=1, uniform="yn_btns")
    btn_bar.columnconfigure(1, weight=1, uniform="yn_btns")
    make_luxury_button(btn_bar, "Yes", choose_yes, padx=26, pady=10, font_size=10, min_width=12).grid(
        row=0, column=0, padx=8, pady=4, sticky="ew"
    )
    make_luxury_button(btn_bar, "No", dlg.destroy, padx=26, pady=10, font_size=10, min_width=12).grid(
        row=0, column=1, padx=8, pady=4, sticky="ew"
    )
    dlg.bind("<Escape>", lambda _e: dlg.destroy())
    dlg.wait_window()
    return bool(answer["yes"])


def ask_directory_half_dialog(
    parent: Any,
    title: str,
    initialdir: str,
    mustexist: bool = False,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Half-size centered folder explorer replacing native filedialog.askdirectory."""
    import tkinter as tk
    from tkinter import ttk

    dlg = open_centered_half_dialog(parent, title)
    chosen = {"path": ""}

    start_path = Path(initialdir).expanduser()
    if not start_path.is_dir():
        start_path = start_path.parent if start_path.parent.is_dir() else (repo_root if repo_root.is_dir() else Path.home())
    cur_dir = {"path": start_path.resolve()}

    outer = ttk.Frame(dlg, padding=12)
    outer.grid(row=0, column=0, sticky="nsew")
    dlg.rowconfigure(0, weight=1)
    dlg.columnconfigure(0, weight=1)
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(1, weight=1)

    nav = ttk.Frame(outer)
    nav.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    nav.columnconfigure(1, weight=1)
    ttk.Label(nav, text="Location:", font=("Ubuntu", 10, "bold")).grid(row=0, column=0, padx=(0, 6))
    dir_var = tk.StringVar(master=dlg, value=str(cur_dir["path"]))
    dir_entry = ttk.Entry(nav, textvariable=dir_var)
    dir_entry.grid(row=0, column=1, sticky="ew", padx=(0, 6))

    list_frame = ttk.Frame(outer)
    list_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
    list_frame.rowconfigure(0, weight=1)
    list_frame.columnconfigure(0, weight=1)

    tree = ttk.Treeview(list_frame, columns=("name",), show="headings", selectmode="browse")
    tree.heading("name", text="Folders (double-click to enter)")
    tree.column("name", anchor="w")
    tree.grid(row=0, column=0, sticky="nsew")
    vsb = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
    vsb.grid(row=0, column=1, sticky="ns")
    tree.configure(yscrollcommand=vsb.set)

    sel_row = ttk.Frame(outer)
    sel_row.grid(row=2, column=0, sticky="ew", pady=(0, 10))
    sel_row.columnconfigure(1, weight=1)
    ttk.Label(sel_row, text="Selected Folder:", font=("Ubuntu", 10, "bold")).grid(row=0, column=0, padx=(0, 6))
    sel_var = tk.StringVar(master=dlg, value=str(cur_dir["path"]))
    ttk.Entry(sel_row, textvariable=sel_var).grid(row=0, column=1, sticky="ew")

    item_paths: Dict[str, Path] = {}

    def populate(target: Path) -> None:
        if not target.is_dir():
            return
        cur_dir["path"] = target.resolve()
        dir_var.set(str(cur_dir["path"]))
        sel_var.set(str(cur_dir["path"]))
        tree.delete(*tree.get_children())
        item_paths.clear()
        try:
            subdirs = sorted(
                (p for p in cur_dir["path"].iterdir() if p.is_dir() and not p.name.startswith(".")),
                key=lambda p: p.name.lower(),
            )
        except OSError:
            subdirs = []
        for sub in subdirs:
            iid = tree.insert("", "end", values=(f"📁  {sub.name}",))
            item_paths[iid] = sub

    def go_entered(*_: Any) -> None:
        cand = Path(dir_var.get().strip()).expanduser()
        if cand.is_dir():
            populate(cand)

    make_luxury_button(nav, "Go", go_entered, padx=14, pady=5).grid(row=0, column=2, padx=4)
    make_luxury_button(nav, "⬆ Up", lambda: populate(cur_dir["path"].parent), padx=14, pady=5).grid(row=0, column=3, padx=4)
    make_luxury_button(nav, "Repo Root", lambda: populate(repo_root), padx=14, pady=5).grid(row=0, column=4, padx=4)

    def on_select(*_: Any) -> None:
        sel = tree.selection()
        if sel and sel[0] in item_paths:
            sel_var.set(str(item_paths[sel[0]]))

    def on_double_click(*_: Any) -> None:
        sel = tree.selection()
        if sel and sel[0] in item_paths:
            populate(item_paths[sel[0]])

    def accept() -> None:
        raw = sel_var.get().strip()
        if not raw:
            return
        cand = Path(raw).expanduser()
        if mustexist and not cand.is_dir():
            show_half_message_dialog(dlg, "Folder Not Found", f"Folder does not exist:\n{cand}", is_error=True)
            return
        chosen["path"] = str(cand)
        dlg.destroy()

    def create_folder() -> None:
        raw = sel_var.get().strip()
        if not raw:
            show_half_message_dialog(dlg, "Invalid Folder", "Enter a folder path in Selected Folder.", is_error=True)
            return
        cand = Path(raw).expanduser()
        if not cand.is_absolute():
            cand = cur_dir["path"] / cand
        try:
            cand.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            show_half_message_dialog(dlg, "Cannot Create Folder", f"{cand}\n{exc}", is_error=True)
            return
        populate(cand)

    dir_entry.bind("<Return>", go_entered)
    tree.bind("<<TreeviewSelect>>", on_select)
    tree.bind("<Double-1>", on_double_click)
    dlg.bind("<Escape>", lambda _e: dlg.destroy())

    btns = ttk.Frame(outer)
    btns.grid(row=3, column=0, sticky="e")
    if not mustexist:
        make_luxury_button(btns, "Create Folder", create_folder, padx=22, pady=8, font_size=10).pack(side="left", padx=6)
    make_luxury_button(btns, "Select Folder", accept, padx=22, pady=8, font_size=10).pack(side="left", padx=6)
    make_luxury_button(btns, "Cancel", dlg.destroy, padx=22, pady=8, font_size=10).pack(side="left", padx=6)

    populate(cur_dir["path"])
    dlg.wait_window()
    return chosen["path"]


def ask_file_half_dialog(
    parent: Any,
    title: str,
    initialdir: str,
    filetypes: Optional[Sequence[Tuple[str, str]]] = None,
    save_mode: bool = False,
    initialfile: str = "",
    defaultextension: str = "",
    repo_root: Path = REPO_ROOT,
) -> str:
    """Half-size centered file explorer (open/save) replacing native filedialog."""
    import tkinter as tk
    from tkinter import ttk

    dlg = open_centered_half_dialog(parent, title)
    chosen = {"path": ""}
    ft_list = list(filetypes) if filetypes else [("All files", "*.*")]

    start_path = Path(initialdir).expanduser()
    if start_path.is_file():
        start_path = start_path.parent
    if not start_path.is_dir():
        start_path = repo_root if repo_root.is_dir() else Path.home()
    cur_dir = {"path": start_path.resolve()}

    outer = ttk.Frame(dlg, padding=12)
    outer.grid(row=0, column=0, sticky="nsew")
    dlg.rowconfigure(0, weight=1)
    dlg.columnconfigure(0, weight=1)
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(1, weight=1)

    nav = ttk.Frame(outer)
    nav.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    nav.columnconfigure(1, weight=1)
    ttk.Label(nav, text="Location:", font=("Ubuntu", 10, "bold")).grid(row=0, column=0, padx=(0, 6))
    dir_var = tk.StringVar(master=dlg, value=str(cur_dir["path"]))
    dir_entry = ttk.Entry(nav, textvariable=dir_var)
    dir_entry.grid(row=0, column=1, sticky="ew", padx=(0, 6))

    list_frame = ttk.Frame(outer)
    list_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
    list_frame.rowconfigure(0, weight=1)
    list_frame.columnconfigure(0, weight=1)

    tree = ttk.Treeview(list_frame, columns=("name", "kind"), show="headings", selectmode="browse")
    tree.heading("name", text="Name (double-click folder to open, file to select)")
    tree.heading("kind", text="Type")
    tree.column("name", anchor="w", width=420)
    tree.column("kind", anchor="w", width=110)
    tree.grid(row=0, column=0, sticky="nsew")
    vsb = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
    vsb.grid(row=0, column=1, sticky="ns")
    tree.configure(yscrollcommand=vsb.set)

    bottom_form = ttk.Frame(outer)
    bottom_form.grid(row=2, column=0, sticky="ew", pady=(0, 10))
    bottom_form.columnconfigure(1, weight=1)

    ttk.Label(bottom_form, text="File Name:", font=("Ubuntu", 10, "bold")).grid(row=0, column=0, padx=(0, 6), pady=3, sticky="w")
    fname_var = tk.StringVar(master=dlg, value=initialfile)
    ttk.Entry(bottom_form, textvariable=fname_var).grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=3)

    filter_labels = [f"{lbl} ({pat})" for lbl, pat in ft_list]
    filter_var = tk.StringVar(master=dlg, value=filter_labels[0])
    filter_combo = ttk.Combobox(bottom_form, textvariable=filter_var, values=filter_labels, state="readonly", width=26)
    filter_combo.grid(row=0, column=2, pady=3)

    item_map: Dict[str, Tuple[str, Path]] = {}

    def active_patterns() -> List[str]:
        idx = filter_labels.index(filter_var.get()) if filter_var.get() in filter_labels else 0
        raw_pat = ft_list[idx][1].strip()
        return [p for p in raw_pat.split() if p] or ["*"]

    def matches_filter(name: str, patterns: Sequence[str]) -> bool:
        if any(p in ("*", "*.*") for p in patterns):
            return True
        return any(fnmatch.fnmatch(name.lower(), p.lower()) for p in patterns)

    def populate(target: Path) -> None:
        if not target.is_dir():
            return
        cur_dir["path"] = target.resolve()
        dir_var.set(str(cur_dir["path"]))
        tree.delete(*tree.get_children())
        item_map.clear()
        pats = active_patterns()
        try:
            entries = list(cur_dir["path"].iterdir())
        except OSError:
            entries = []
        subdirs = sorted((p for p in entries if p.is_dir() and not p.name.startswith(".")), key=lambda p: p.name.lower())
        files = sorted(
            (p for p in entries if p.is_file() and not p.name.startswith(".") and matches_filter(p.name, pats)),
            key=lambda p: p.name.lower(),
        )
        for sub in subdirs:
            iid = tree.insert("", "end", values=(f"📁  {sub.name}", "Folder"))
            item_map[iid] = ("dir", sub)
        for fpath in files:
            iid = tree.insert("", "end", values=(f"📄  {fpath.name}", fpath.suffix or "File"))
            item_map[iid] = ("file", fpath)

    def go_entered(*_: Any) -> None:
        cand = Path(dir_var.get().strip()).expanduser()
        if cand.is_dir():
            populate(cand)

    make_luxury_button(nav, "Go", go_entered, padx=14, pady=5).grid(row=0, column=2, padx=4)
    make_luxury_button(nav, "⬆ Up", lambda: populate(cur_dir["path"].parent), padx=14, pady=5).grid(row=0, column=3, padx=4)
    make_luxury_button(nav, "Repo Root", lambda: populate(repo_root), padx=14, pady=5).grid(row=0, column=4, padx=4)

    def accept() -> None:
        raw_name = fname_var.get().strip()
        if not raw_name:
            return
        cand = Path(raw_name).expanduser()
        if not cand.is_absolute():
            cand = cur_dir["path"] / cand
        if save_mode and defaultextension and not cand.suffix:
            cand = Path(str(cand) + defaultextension)
        if not save_mode and not cand.is_file():
            show_half_message_dialog(dlg, "File Not Found", f"Selected file does not exist:\n{cand}", is_error=True)
            return
        chosen["path"] = str(cand.resolve())
        dlg.destroy()

    def on_select(*_: Any) -> None:
        sel = tree.selection()
        if sel and sel[0] in item_map:
            kind, p = item_map[sel[0]]
            if kind == "file":
                fname_var.set(p.name)

    def on_double_click(*_: Any) -> None:
        sel = tree.selection()
        if not sel or sel[0] not in item_map:
            return
        kind, p = item_map[sel[0]]
        if kind == "dir":
            populate(p)
        else:
            fname_var.set(p.name)
            accept()

    dir_entry.bind("<Return>", go_entered)
    filter_combo.bind("<<ComboboxSelected>>", lambda *_: populate(cur_dir["path"]))
    tree.bind("<<TreeviewSelect>>", on_select)
    tree.bind("<Double-1>", on_double_click)
    dlg.bind("<Escape>", lambda _e: dlg.destroy())

    btns = ttk.Frame(outer)
    btns.grid(row=3, column=0, sticky="e")
    make_luxury_button(
        btns,
        "Save" if save_mode else "Open",
        accept,
        padx=24,
        pady=8,
        font_size=10,
    ).pack(side="left", padx=6)
    make_luxury_button(btns, "Cancel", dlg.destroy, padx=24, pady=8, font_size=10).pack(side="left", padx=6)

    populate(cur_dir["path"])
    dlg.wait_window()
    return chosen["path"]


def _build_spyder_grid_panel(
    parent_widget: Any,
    initial_model: GridTableModel,
    on_close: Callable[[], None],
    open_popup_cb: Optional[Callable[[GridTableModel], None]] = None,
) -> Dict[str, Any]:
    """Build an interactive Spyder-style DataFrame Canvas grid with Filter, Transpose, Cols & Sub-Data stack."""
    import tkinter as tk
    from tkinter import font as tkfont, ttk

    header_font = tkfont.Font(root=parent_widget, family="Ubuntu", size=11, weight="bold")
    filter_font = tkfont.Font(root=parent_widget, family="Ubuntu", size=10)
    cell_font = tkfont.Font(root=parent_widget, family="Ubuntu Mono", size=10)
    index_font = tkfont.Font(root=parent_widget, family="Ubuntu Mono", size=10, weight="bold")

    history: List[Tuple[str, GridTableModel]] = [(initial_model.title, initial_model)]
    state: Dict[str, Any] = {
        "model": initial_model,
        "row_offset": 0,
        "col_offset": 0,
        "col_widths": {},
        "default_col_w": 128,
        "index_col_w": 92,
        "header_h": max(44, header_font.metrics("linespace") + 12),
        "filter_h": max(34, filter_font.metrics("linespace") + 10),
        "row_h": 25,
        "fmt_idx": 0,
        "selected": (0, 0),
        "drag_col": None,
        "drag_start_x": 0,
        "drag_start_w": 128,
        "filter_col_editing": None,
    }

    bg_color_var = tk.BooleanVar(master=parent_widget, value=True)
    col_minmax_var = tk.BooleanVar(master=parent_widget, value=True)
    slice_var = tk.StringVar(master=parent_widget, value="")
    title_var = tk.StringVar(master=parent_widget, value="")
    subtitle_var = tk.StringVar(master=parent_widget, value="")
    status_var = tk.StringVar(master=parent_widget, value="")
    dims_var = tk.StringVar(master=parent_widget, value="")

    container = ttk.Frame(parent_widget)
    container.grid(row=0, column=0, sticky="nsew")
    parent_widget.rowconfigure(0, weight=1)
    parent_widget.columnconfigure(0, weight=1)
    container.columnconfigure(0, weight=1)
    container.rowconfigure(2, weight=1)

    # Row 0: Breadcrumb & Sub-Data Navigation Bar
    nav_bar = ttk.Frame(container, padding=(8, 6))
    nav_bar.grid(row=0, column=0, sticky="ew")
    nav_bar.columnconfigure(2, weight=1)

    back_btn = make_luxury_button(nav_bar, text="⬅ Back", enabled=False, padx=16, pady=8, font_size=10, min_width=10)
    back_btn.grid(row=0, column=0, padx=6, pady=4)
    root_btn = make_luxury_button(nav_bar, text="🏠 Root", enabled=False, padx=16, pady=8, font_size=10, min_width=10)
    root_btn.grid(row=0, column=1, padx=6, pady=4)

    title_lbl = ttk.Label(nav_bar, textvariable=title_var, font=("Ubuntu", 11, "bold"), foreground=CP1_LABEL_FG)
    title_lbl.grid(row=0, column=2, sticky="w", padx=8)

    if open_popup_cb is not None:
        make_luxury_button(
            nav_bar,
            text="🗗 Pop Out Window",
            command=lambda: open_popup_cb(state["model"]),
            padx=18,
            pady=8,
            font_size=10,
            min_width=16,
        ).grid(row=0, column=3, padx=6, pady=4)

    # Row 1: Row/Column Switching Toolbar (Transpose, Columns, Slice, Clear)
    rc_bar = ttk.Frame(container, padding=(8, 4))
    rc_bar.grid(row=1, column=0, sticky="ew")
    rc_bar.columnconfigure(6, weight=1)

    transpose_btn = make_luxury_button(
        rc_bar,
        text="⇄ Swap Rows ↔ Cols (Transpose)",
        padx=18,
        pady=8,
        font_size=10,
    )
    transpose_btn.grid(row=0, column=0, padx=6, pady=4)

    cols_btn = make_luxury_button(rc_bar, text="Columns (Select / Reorder)...", padx=18, pady=8, font_size=10)
    cols_btn.grid(row=0, column=1, padx=6, pady=4)

    ttk.Label(rc_bar, text="Row Slice (start:stop:step):", foreground=CP1_LABEL_FG).grid(row=0, column=2, padx=6)
    slice_entry = ttk.Entry(rc_bar, textvariable=slice_var, width=14)
    slice_entry.grid(row=0, column=3, padx=6)
    apply_slice_btn = make_luxury_button(rc_bar, text="Apply", padx=16, pady=8, font_size=10, min_width=9)
    apply_slice_btn.grid(row=0, column=4, padx=6, pady=4)
    reset_btn = make_luxury_button(rc_bar, text="Reset View / Filters", padx=18, pady=8, font_size=10)
    reset_btn.grid(row=0, column=5, padx=6, pady=4)

    ttk.Label(rc_bar, textvariable=subtitle_var, foreground=CP1_LABEL_FG).grid(row=0, column=6, sticky="w", padx=6)

    # Row 2: Spyder Virtualized Canvas + Vertical & Horizontal Scrollbars
    grid_frame = ttk.Frame(container)
    grid_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
    grid_frame.rowconfigure(0, weight=1)
    grid_frame.columnconfigure(0, weight=1)

    canvas = tk.Canvas(grid_frame, background=CP1_SURFACE_BG, highlightthickness=1, highlightbackground=CP1_BORDER)
    canvas.grid(row=0, column=0, sticky="nsew")
    v_scroll = ttk.Scrollbar(grid_frame, orient="vertical")
    v_scroll.grid(row=0, column=1, sticky="ns")
    h_scroll = ttk.Scrollbar(grid_frame, orient="horizontal")
    h_scroll.grid(row=1, column=0, sticky="ew")

    filter_entry = tk.Entry(
        canvas,
        background=CP1_SURFACE_BG,
        foreground=CP1_VALUE_FG,
        insertbackground=CP1_BTN_BG,
        relief="solid",
        borderwidth=1,
        font=("Ubuntu Mono", 10),
    )

    # Row 3: Selected Cell Readout
    status_frame = ttk.Frame(container, padding=(8, 4))
    status_frame.grid(row=3, column=0, sticky="ew")
    status_frame.columnconfigure(0, weight=1)
    ttk.Label(status_frame, textvariable=status_var, foreground=CP1_LABEL_FG).grid(row=0, column=0, sticky="w")
    ttk.Label(status_frame, textvariable=dims_var, font=("Ubuntu Mono", 10, "bold"), foreground=CP1_LABEL_FG).grid(
        row=0, column=1, sticky="e"
    )

    # Row 4: Spyder Bottom Control Bar
    bottom_bar = ttk.Frame(container, padding=(8, 8))
    bottom_bar.grid(row=4, column=0, sticky="ew")
    bottom_bar.columnconfigure(5, weight=1)

    fmt_btn = make_luxury_button(bottom_bar, text="Format (%.6g)", padx=18, pady=8, font_size=10, min_width=14)
    fmt_btn.grid(row=0, column=0, padx=6, pady=4)
    resize_btn = make_luxury_button(bottom_bar, text="Resize", padx=18, pady=8, font_size=10, min_width=10)
    resize_btn.grid(row=0, column=1, padx=6, pady=4)

    bg_chk = ttk.Checkbutton(bottom_bar, text="Background color", variable=bg_color_var)
    bg_chk.grid(row=0, column=2, padx=(8, 12))
    col_mm_chk = ttk.Checkbutton(bottom_bar, text="Column min/max", variable=col_minmax_var)
    col_mm_chk.grid(row=0, column=3, padx=(0, 12))

    drill_btn = make_luxury_button(bottom_bar, text="🔍 Enter Sub-Data (Double-Click)", padx=18, pady=8, font_size=10)
    drill_btn.grid(row=0, column=4, padx=6, pady=4)

    ok_btn = make_luxury_button(bottom_bar, text="OK", command=on_close, padx=22, pady=8, font_size=10, min_width=10)
    ok_btn.grid(row=0, column=6, padx=6, pady=4)
    cancel_btn = make_luxury_button(bottom_bar, text="Cancel", command=on_close, padx=22, pady=8, font_size=10, min_width=10)
    cancel_btn.grid(row=0, column=7, padx=6, pady=4)

    def _col_width(disp_c: int) -> int:
        return int(state["col_widths"].get(disp_c, state["default_col_w"]))

    def _hide_filter_editor(commit: bool = True) -> None:
        edit_c = state["filter_col_editing"]
        if edit_c is None:
            return
        state["filter_col_editing"] = None
        text_val = filter_entry.get()
        filter_entry.place_forget()
        if commit:
            model: GridTableModel = state["model"]
            model.set_filter(int(edit_c), text_val)
            state["row_offset"] = 0
            redraw()

    def _clamp_offsets() -> None:
        model: GridTableModel = state["model"]
        n_rows, n_cols = model.shape()
        cw = max(1, canvas.winfo_width())
        ch = max(1, canvas.winfo_height())
        usable_h = max(state["row_h"], ch - state["header_h"] - state["filter_h"])
        vis_rows = max(1, usable_h // state["row_h"])
        state["row_offset"] = max(0, min(int(state["row_offset"]), max(0, n_rows - vis_rows)))
        remaining_w = max(1, cw - state["index_col_w"])
        last_start = n_cols
        for col in range(n_cols - 1, -1, -1):
            col_w = _col_width(col)
            if col_w > remaining_w:
                break
            remaining_w -= col_w
            last_start = col
        max_col_offset = max(0, min(last_start, n_cols - 1))
        state["col_offset"] = max(0, min(int(state["col_offset"]), max_col_offset))

    def fit_header(text: str, width: int) -> str:
        if header_font.measure(text) <= width:
            return text
        suffix = "…"
        if header_font.measure(suffix) > width:
            return ""
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if header_font.measure(text[:mid] + suffix) <= width:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + suffix

    def redraw() -> None:
        canvas.delete("all")
        model: GridTableModel = state["model"]
        n_rows, n_cols = model.shape()
        _clamp_offsets()

        crumbs = "  ›  ".join(lbl for lbl, _ in history[-4:])
        title_var.set(crumbs)
        subtitle_var.set(model.subtitle)
        orient = "Transposed (Features as Rows)" if model.transposed else "Standard (Samples as Rows)"
        dims_var.set(f"{orient} | {n_rows:,} rows × {n_cols:,} cols")
        set_luxury_button_state(back_btn, len(history) > 1)
        set_luxury_button_state(root_btn, len(history) > 1)

        cw = max(1, canvas.winfo_width())
        ch = max(1, canvas.winfo_height())
        idx_w = state["index_col_w"]
        hdr_h = state["header_h"]
        flt_h = state["filter_h"]
        top_h = hdr_h + flt_h
        row_h = state["row_h"]
        r0 = int(state["row_offset"])
        c0 = int(state["col_offset"])

        vis_rows = max(1, (ch - top_h) // row_h + 1)
        r_end = min(n_rows, r0 + vis_rows)

        # Update scrollbars
        if n_rows <= 0:
            v_scroll.set(0.0, 1.0)
        else:
            v_scroll.set(r0 / n_rows, min(1.0, r_end / n_rows))

        # Top-left Index corner + Filter row stub
        canvas.create_rectangle(0, 0, idx_w, hdr_h, fill=CP1_BG, outline=CP1_BORDER)
        canvas.create_text(idx_w // 2, hdr_h // 2, text="Index", fill=CP1_LABEL_FG, font=header_font)
        canvas.create_rectangle(0, hdr_h, idx_w, top_h, fill=CP1_BG, outline=CP1_BORDER)

        # Determine visible columns across viewport width
        vis_cols_spec: List[Tuple[int, int, int]] = []
        x_cur = idx_w
        c = c0
        while c < n_cols and x_cur < cw:
            w = _col_width(c)
            vis_cols_spec.append((c, x_cur, x_cur + w))
            x_cur += w
            c += 1

        if n_cols <= 0:
            h_scroll.set(0.0, 1.0)
        else:
            c_end = vis_cols_spec[-1][0] + 1 if vis_cols_spec else c0 + 1
            h_scroll.set(c0 / n_cols, min(1.0, c_end / n_cols))

        # Draw column headers & inline Filter boxes
        for disp_c, x1, x2 in vis_cols_spec:
            hdr_text = model.disp_col_header(disp_c)
            canvas.create_rectangle(x1, 0, x2, hdr_h, fill=CP1_BG, outline=CP1_BORDER)
            canvas.create_text(
                (x1 + x2) // 2,
                hdr_h // 2,
                text=fit_header(hdr_text, max(0, x2 - x1 - 16)),
                fill=CP1_LABEL_FG,
                font=header_font,
            )
            f_txt = model.get_filter(disp_c)
            f_bg = "#2C3959" if f_txt else CP1_SURFACE_BG
            canvas.create_rectangle(x1 + 1, hdr_h + 2, x2 - 1, top_h - 2, fill=f_bg, outline=CP1_BORDER)
            canvas.create_text(
                x1 + 6,
                hdr_h + flt_h // 2,
                anchor="w",
                text=f_txt if f_txt else "Filter",
                fill=CP1_BTN_BG if f_txt else CP1_LABEL_FG,
                font=filter_font,
            )

        # Draw visible rows (Index label + Spyder colored cells)
        float_fmt = FLOAT_FORMATS[state["fmt_idx"]]
        bg_on = bool(bg_color_var.get())
        cmm_on = bool(col_minmax_var.get())
        sel_r, sel_c = state["selected"]

        if n_rows == 0 or n_cols == 0:
            canvas.create_text(
                cw // 2,
                ch // 2,
                text="No rows match the current filter / slice.",
                fill=CP1_LABEL_FG,
                font=("Ubuntu", 11),
            )
            status_var.set("Empty view — click 'Reset View / Filters' to restore.")
            return

        for disp_r in range(r0, r_end):
            y1 = top_h + (disp_r - r0) * row_h
            y2 = y1 + row_h
            row_lbl = model.disp_row_label(disp_r)
            idx_bg = "#2C3959" if disp_r == sel_r else CP1_BG
            canvas.create_rectangle(0, y1, idx_w, y2, fill=idx_bg, outline=CP1_BORDER)
            canvas.create_text(
                8,
                (y1 + y2) // 2,
                anchor="w",
                text=row_lbl,
                fill=CP1_BTN_BG if disp_r == sel_r else CP1_LABEL_FG,
                font=("Ubuntu Mono", 10, "bold" if disp_r == sel_r else "normal"),
            )

            for disp_c, x1, x2 in vis_cols_spec:
                val, _, _ = model.cell_info(disp_r, disp_c)
                raw_bg = model.cell_bg_color(disp_r, disp_c, bg_enabled=bg_on, col_minmax=cmm_on)
                is_spyder_heat = raw_bg != "#ffffff"
                cell_bg = raw_bg if is_spyder_heat else CP1_SURFACE_BG
                canvas.create_rectangle(x1, y1, x2, y2, fill=cell_bg, outline=CP1_BORDER)
                txt = _format_cell_value(val, float_fmt)
                is_link = isinstance(val, str) and val.startswith("▶")
                text_fill = (
                    ("#111827" if is_spyder_heat else CP1_BTN_BG)
                    if is_link
                    else ("#111827" if is_spyder_heat else CP1_VALUE_FG)
                )
                canvas.create_text(
                    x1 + 6,
                    (y1 + y2) // 2,
                    anchor="w",
                    text=txt,
                    fill=text_fill,
                    font=("Ubuntu Mono", 10, "bold" if is_link else "normal"),
                )
                if disp_r == sel_r and disp_c == sel_c:
                    canvas.create_rectangle(x1 + 1, y1 + 1, x2 - 1, y2 - 1, outline=CP1_BTN_BG, width=2)

        if 0 <= sel_r < n_rows and 0 <= sel_c < n_cols:
            status_var.set(model.column_detail_summary(sel_r, sel_c))

    def load_model(new_model: GridTableModel, push_history: bool = True, crumb_label: Optional[str] = None) -> None:
        _hide_filter_editor(commit=False)
        if push_history:
            history.append((crumb_label or new_model.title, new_model))
        else:
            history[:] = [(crumb_label or new_model.title, new_model)]
        state["model"] = new_model
        state["row_offset"] = 0
        state["col_offset"] = 0
        state["col_widths"] = {}
        state["selected"] = (0, 0)
        slice_var.set(new_model.slice_spec)
        auto_resize_columns(redraw_after=False)
        redraw()

    def go_back() -> None:
        if len(history) > 1:
            _hide_filter_editor(commit=False)
            history.pop()
            _, prev_model = history[-1]
            state["model"] = prev_model
            state["row_offset"] = 0
            state["col_offset"] = 0
            state["col_widths"] = {}
            state["selected"] = (0, 0)
            slice_var.set(prev_model.slice_spec)
            auto_resize_columns(redraw_after=False)
            redraw()

    def go_root() -> None:
        if len(history) > 1:
            _hide_filter_editor(commit=False)
            del history[1:]
            _, root_model = history[0]
            state["model"] = root_model
            state["row_offset"] = 0
            state["col_offset"] = 0
            state["col_widths"] = {}
            state["selected"] = (0, 0)
            slice_var.set(root_model.slice_spec)
            auto_resize_columns(redraw_after=False)
            redraw()

    def auto_resize_columns(redraw_after: bool = True) -> None:
        model: GridTableModel = state["model"]
        n_rows, n_cols = model.shape()
        float_fmt = FLOAT_FORMATS[state["fmt_idx"]]
        if n_rows > 0:
            sample_r = [min(n_rows - 1, i) for i in (0, 1, 2, n_rows // 2, n_rows - 1)]
            max_idx_width = max(index_font.measure(model.disp_row_label(r)) for r in sample_r)
            state["index_col_w"] = min(320, max(96, header_font.measure("Index") + 24, max_idx_width + 20))
        else:
            state["index_col_w"] = 84

        max_scan_cols = min(n_cols, 60)
        new_widths: Dict[int, int] = {}
        for c in range(max_scan_cols):
            hdr_width = header_font.measure(model.disp_col_header(c) + " ▼")
            val_width = cell_font.measure("00000000")
            for r in range(min(n_rows, 15)):
                v, _, _ = model.cell_info(r, c)
                val_width = max(val_width, cell_font.measure(_format_cell_value(v, float_fmt)))
            new_widths[c] = min(640, max(140, hdr_width + 24, val_width + 20))
        state["col_widths"] = new_widths
        if redraw_after:
            redraw()

    def cycle_format() -> None:
        state["fmt_idx"] = (int(state["fmt_idx"]) + 1) % len(FLOAT_FORMATS)
        fmt_btn.configure(text=f"Format ({FLOAT_FORMATS[state['fmt_idx']]})")
        redraw()

    def on_transpose() -> None:
        _hide_filter_editor(commit=False)
        model: GridTableModel = state["model"]
        model.toggle_transpose()
        state["row_offset"] = 0
        state["col_offset"] = 0
        state["selected"] = (0, 0)
        auto_resize_columns(redraw_after=True)

    def apply_slice() -> None:
        model: GridTableModel = state["model"]
        try:
            model.set_slice_spec(slice_var.get())
        except ValueError as exc:
            show_half_message_dialog(container.winfo_toplevel(), "Invalid Row Slice", str(exc), is_error=True)
            return
        state["row_offset"] = 0
        state["selected"] = (0, 0)
        redraw()

    def reset_view() -> None:
        _hide_filter_editor(commit=False)
        model: GridTableModel = state["model"]
        model.clear_filters_and_slice()
        slice_var.set("")
        state["row_offset"] = 0
        state["col_offset"] = 0
        state["selected"] = (0, 0)
        redraw()

    def open_columns_dialog() -> None:
        model: GridTableModel = state["model"]
        dlg = open_centered_half_dialog(container.winfo_toplevel(), "Select & Reorder Columns")

        frm = ttk.Frame(dlg, padding=10)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Check columns to display and use Move Up / Down to reorder:").pack(anchor="w", pady=(0, 6))

        preset_bar = ttk.Frame(frm)
        preset_bar.pack(fill="x", pady=(0, 6))

        order_list: List[int] = list(model.visible_col_indices) + [
            i for i in range(len(model.all_col_names)) if i not in model.visible_col_indices
        ]
        checked: Dict[int, tk.BooleanVar] = {
            i: tk.BooleanVar(master=dlg, value=(i in model.visible_col_indices))
            for i in range(len(model.all_col_names))
        }

        list_frame = ttk.Frame(frm)
        list_frame.pack(fill="both", expand=True)
        lb = tk.Listbox(
            list_frame,
            selectmode="browse",
            font=("Ubuntu Mono", 10),
            background=CP1_SURFACE_BG,
            foreground=CP1_VALUE_FG,
            selectbackground=CP1_BTN_BG,
            selectforeground=CP1_BTN_FG,
            highlightthickness=0,
            borderwidth=1,
            relief="solid",
        )
        lb.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.configure(yscrollcommand=sb.set)

        def refresh_lb(sel_pos: int = 0) -> None:
            lb.delete(0, "end")
            for raw_i in order_list:
                mark = "[x]" if checked[raw_i].get() else "[ ]"
                nm = model.all_col_names[raw_i]
                tag = model.col_tags.get(nm, "")
                tag_str = f" ({tag})" if tag else ""
                lb.insert("end", f"{mark} {nm}{tag_str}")
            if order_list:
                sel_pos = max(0, min(sel_pos, len(order_list) - 1))
                lb.selection_set(sel_pos)
                lb.see(sel_pos)

        def set_by_tag(target_tags: Optional[set]) -> None:
            for raw_i in order_list:
                nm = model.all_col_names[raw_i]
                checked[raw_i].set(target_tags is None or model.col_tags.get(nm) in target_tags)
            refresh_lb()

        make_luxury_button(preset_bar, "All", lambda: set_by_tag(None), padx=14, pady=4).pack(side="left", padx=4)
        make_luxury_button(preset_bar, "Obs (13)", lambda: set_by_tag({"OBS"}), padx=14, pady=4).pack(side="left", padx=4)
        make_luxury_button(preset_bar, "Act (4)", lambda: set_by_tag({"ACT"}), padx=14, pady=4).pack(side="left", padx=4)
        make_luxury_button(preset_bar, "Obs+Act", lambda: set_by_tag({"OBS", "ACT"}), padx=14, pady=4).pack(side="left", padx=4)
        make_luxury_button(preset_bar, "PX4", lambda: set_by_tag({"PX4", "STAT"}), padx=14, pady=4).pack(side="left", padx=4)

        def toggle_selected(*_: Any) -> None:
            sel = lb.curselection()
            if not sel:
                return
            pos = int(sel[0])
            raw_i = order_list[pos]
            checked[raw_i].set(not checked[raw_i].get())
            refresh_lb(pos)

        def move_item(delta: int) -> None:
            sel = lb.curselection()
            if not sel:
                return
            pos = int(sel[0])
            nxt = pos + delta
            if 0 <= nxt < len(order_list):
                order_list[pos], order_list[nxt] = order_list[nxt], order_list[pos]
                refresh_lb(nxt)

        lb.bind("<Double-Button-1>", toggle_selected)
        lb.bind("<space>", toggle_selected)

        act_bar = ttk.Frame(frm)
        act_bar.pack(fill="x", pady=(8, 0))
        make_luxury_button(act_bar, "Toggle [x]", toggle_selected, padx=14, pady=5).pack(side="left", padx=4)
        make_luxury_button(act_bar, "▲ Up", lambda: move_item(-1), padx=14, pady=5).pack(side="left", padx=4)
        make_luxury_button(act_bar, "▼ Down", lambda: move_item(1), padx=14, pady=5).pack(side="left", padx=4)

        def apply_cols() -> None:
            chosen = [i for i in order_list if checked[i].get()]
            if not chosen:
                show_half_message_dialog(dlg, "No Columns", "Select at least one column.", is_error=True)
                return
            model.set_visible_columns(chosen)
            state["col_offset"] = 0
            auto_resize_columns(redraw_after=True)
            dlg.destroy()

        make_luxury_button(act_bar, "Cancel", dlg.destroy, padx=18, pady=5).pack(side="right", padx=4)
        make_luxury_button(act_bar, "Apply", apply_cols, padx=18, pady=5).pack(side="right", padx=4)
        refresh_lb()

    def _hit_column_at_x(x: int) -> Tuple[Optional[int], Optional[int], int, int]:
        """Return (disp_c, border_disp_c, x1, x2) at canvas coordinate x."""
        model: GridTableModel = state["model"]
        _, n_cols = model.shape()
        idx_w = state["index_col_w"]
        if x < idx_w:
            return None, None, 0, idx_w
        x_cur = idx_w
        c = int(state["col_offset"])
        cw = max(1, canvas.winfo_width())
        while c < n_cols and x_cur < cw:
            w = _col_width(c)
            x_next = x_cur + w
            if abs(x - x_next) <= 4:
                return c, c, x_cur, x_next
            if x_cur <= x < x_next:
                return c, None, x_cur, x_next
            x_cur = x_next
            c += 1
        return None, None, x_cur, x_cur

    def on_canvas_click(event: Any) -> None:
        _hide_filter_editor(commit=True)
        canvas.focus_set()
        model: GridTableModel = state["model"]
        n_rows, _ = model.shape()
        hdr_h = state["header_h"]
        top_h = hdr_h + state["filter_h"]

        disp_c, border_c, x1, x2 = _hit_column_at_x(int(event.x))
        if event.y <= hdr_h:
            if border_c is not None:
                state["drag_col"] = border_c
                state["drag_start_x"] = int(event.x)
                state["drag_start_w"] = _col_width(border_c)
            elif disp_c is not None:
                model.toggle_sort_column(disp_c)
                redraw()
            return

        if hdr_h < event.y <= top_h and disp_c is not None:
            state["filter_col_editing"] = disp_c
            filter_entry.delete(0, "end")
            filter_entry.insert(0, model.get_filter(disp_c))
            filter_entry.place(x=x1 + 1, y=hdr_h + 2, width=max(40, x2 - x1 - 2), height=state["filter_h"] - 4)
            filter_entry.focus_set()
            filter_entry.select_range(0, "end")
            return

        if event.y > top_h and n_rows > 0:
            disp_r = int(state["row_offset"]) + (int(event.y) - top_h) // state["row_h"]
            if 0 <= disp_r < n_rows:
                sel_c = disp_c if disp_c is not None else state["selected"][1]
                state["selected"] = (disp_r, max(0, sel_c))
                redraw()

    def on_canvas_drag(event: Any) -> None:
        drag_c = state["drag_col"]
        if drag_c is None:
            return
        new_w = max(64, min(1200, int(state["drag_start_w"]) + (int(event.x) - int(state["drag_start_x"]))))
        state["col_widths"][drag_c] = new_w
        redraw()

    def on_canvas_release(*_: Any) -> None:
        state["drag_col"] = None

    def trigger_selected_drill() -> None:
        model: GridTableModel = state["model"]
        sel_r, sel_c = state["selected"]
        drilled = model.drill_down(sel_r, sel_c)
        if drilled is not None:
            crumb, sub_model = drilled
            load_model(sub_model, push_history=True, crumb_label=crumb)

    def on_canvas_double_click(event: Any) -> None:
        top_h = state["header_h"] + state["filter_h"]
        if event.y <= top_h:
            return
        model: GridTableModel = state["model"]
        n_rows, n_cols = model.shape()
        disp_r = int(state["row_offset"]) + (int(event.y) - top_h) // state["row_h"]
        disp_c, _, _, _ = _hit_column_at_x(int(event.x))
        if 0 <= disp_r < n_rows:
            c_idx = disp_c if disp_c is not None else min(max(0, state["selected"][1]), max(0, n_cols - 1))
            state["selected"] = (disp_r, c_idx)
            trigger_selected_drill()

    def on_vscroll(*args: Any) -> None:
        _hide_filter_editor(commit=True)
        model: GridTableModel = state["model"]
        n_rows, _ = model.shape()
        if n_rows <= 0:
            return
        if args[0] == "moveto":
            state["row_offset"] = int(float(args[1]) * n_rows)
        elif args[0] == "scroll":
            step = int(args[1])
            mult = 15 if args[2] == "pages" else 3
            state["row_offset"] = int(state["row_offset"]) + step * mult
        redraw()

    def on_hscroll(*args: Any) -> None:
        _hide_filter_editor(commit=True)
        model: GridTableModel = state["model"]
        _, n_cols = model.shape()
        if n_cols <= 0:
            return
        if args[0] == "moveto":
            state["col_offset"] = int(float(args[1]) * n_cols)
        elif args[0] == "scroll":
            step = int(args[1])
            mult = 5 if args[2] == "pages" else 1
            state["col_offset"] = int(state["col_offset"]) + step * mult
        redraw()

    def scroll_rows(delta: int) -> None:
        _hide_filter_editor(commit=True)
        state["row_offset"] = int(state["row_offset"]) + delta
        redraw()

    def scroll_cols(delta: int) -> None:
        _hide_filter_editor(commit=True)
        state["col_offset"] = int(state["col_offset"]) + delta
        redraw()

    v_scroll.configure(command=on_vscroll)
    h_scroll.configure(command=on_hscroll)
    back_btn.configure(command=go_back)
    root_btn.configure(command=go_root)
    transpose_btn.configure(command=on_transpose)
    cols_btn.configure(command=open_columns_dialog)
    apply_slice_btn.configure(command=apply_slice)
    slice_entry.bind("<Return>", lambda *_: apply_slice())
    reset_btn.configure(command=reset_view)
    fmt_btn.configure(command=cycle_format)
    resize_btn.configure(command=lambda: auto_resize_columns(redraw_after=True))
    bg_chk.configure(command=redraw)
    col_mm_chk.configure(command=redraw)
    drill_btn.configure(command=trigger_selected_drill)

    filter_entry.bind("<Return>", lambda *_: _hide_filter_editor(commit=True))
    filter_entry.bind("<Escape>", lambda *_: _hide_filter_editor(commit=False))

    canvas.bind("<Configure>", lambda *_: redraw())
    canvas.bind("<Button-1>", on_canvas_click)
    canvas.bind("<B1-Motion>", on_canvas_drag)
    canvas.bind("<ButtonRelease-1>", on_canvas_release)
    canvas.bind("<Double-Button-1>", on_canvas_double_click)
    canvas.bind("<Return>", lambda *_: trigger_selected_drill())
    canvas.bind("<BackSpace>", lambda *_: go_back())
    canvas.bind("<Button-4>", lambda *_: scroll_rows(-4))
    canvas.bind("<Button-5>", lambda *_: scroll_rows(4))
    canvas.bind("<Shift-Button-4>", lambda *_: scroll_cols(-2))
    canvas.bind("<Shift-Button-5>", lambda *_: scroll_cols(2))
    canvas.bind("<MouseWheel>", lambda e: scroll_rows(-4 if e.delta > 0 else 4))
    canvas.bind("<Shift-MouseWheel>", lambda e: scroll_cols(-2 if e.delta > 0 else 2))

    auto_resize_columns(redraw_after=False)
    return {"frame": container, "load_model": load_model, "redraw": redraw}


def _build_spyder_sidebar(parent: Any) -> Tuple[Any, Any, Any]:
    """Create the shared resizable sidebar and its adjacent table frame."""
    from tkinter import font as tkfont, ttk

    ttk.Style(parent).configure("Sash", sashthickness=14, background=CP1_LABEL_FG)
    paned = ttk.PanedWindow(parent, orient="horizontal")
    paned.bind(
        "<Motion>",
        lambda event: paned.configure(cursor="sb_h_double_arrow" if paned.identify(event.x, event.y) != "" else ""),
    )
    paned.bind("<Leave>", lambda _event: paned.configure(cursor=""))

    left_frame = ttk.LabelFrame(paned, text="Variable & Sub-Data Explorer", padding=8)
    left_frame.rowconfigure(0, weight=1)
    left_frame.columnconfigure(0, weight=1)

    tree_font = tkfont.Font(root=parent, family="Ubuntu", size=11)
    tree_style = ttk.Style(parent)
    tree_style.configure(
        "Spyder.Treeview",
        font=("Ubuntu", 11),
        rowheight=max(36, tree_font.metrics("linespace") + 10),
    )
    tree_style.configure("Spyder.Treeview.Heading", font=("Ubuntu", 11, "bold"), padding=(10, 8))
    tree = ttk.Treeview(
        left_frame, columns=("space", "shape"), show="tree headings", selectmode="browse",
        style="Spyder.Treeview",
    )
    tree.heading("#0", text="Object / Sub-Data")
    tree.heading("space", text="Space / Type")
    tree.heading("shape", text="Shape")
    tree.column("#0", width=max(260, tree_font.measure("Dataset Metadata & Contract") + 48), minwidth=200, stretch=False, anchor="w")
    tree.column("space", width=max(140, tree_font.measure("Two-Tier Split") + 24), minwidth=120, stretch=False, anchor="w")
    tree.column("shape", width=max(110, tree_font.measure("(18,972, 13)") + 24), minwidth=100, stretch=False, anchor="e")
    tree.grid(row=0, column=0, sticky="nsew")

    tree_vsb = ttk.Scrollbar(left_frame, orient="vertical", command=tree.yview)
    tree_vsb.grid(row=0, column=1, sticky="ns")
    tree_hsb = ttk.Scrollbar(left_frame, orient="horizontal", command=tree.xview)
    tree_hsb.grid(row=1, column=0, sticky="ew")
    tree.configure(yscrollcommand=tree_vsb.set, xscrollcommand=tree_hsb.set)

    right_frame = ttk.Frame(paned)
    paned.add(left_frame, weight=1)
    paned.add(right_frame, weight=4)

    def set_initial_sidebar_width(event: Any) -> None:
        if event.width > 1:
            paned.sashpos(0, int(event.width * 0.32))
            paned.unbind("<Configure>")

    paned.bind("<Configure>", set_initial_sidebar_width)

    return paned, tree, right_frame


def show_spyder_table(
    data: Any,
    title: str = "DataFrame",
    parent: Optional[Any] = None,
    maximized: bool = True,
    *,
    sidebar: Optional[Dict[str, GridTableModel]] = None,
) -> None:
    """Display a Spyder-style table, optionally with a resizable section sidebar."""
    import tkinter as tk

    model = data if isinstance(data, GridTableModel) else GridTableModel.from_nested_object(title, data)
    owns_root = parent is None
    if owns_root:
        ensure_repo_root_cwd(REPO_ROOT)
        win: Any = tk.Tk()
        win.title(f"{model.title} - DataFrame")
        _maximize_tk_window(win)
    elif maximized:
        win = tk.Toplevel(parent)
        win.title(f"{model.title} - DataFrame")
        _maximize_tk_window(win)
    else:
        win = open_centered_half_dialog(parent, f"{model.title} - DataFrame")

    def open_popup(sub_m: GridTableModel) -> None:
        show_spyder_table(sub_m, title=sub_m.title, parent=win, maximized=False)

    if sidebar is None:
        _build_spyder_grid_panel(win, model, on_close=win.destroy, open_popup_cb=open_popup)
    else:
        win.rowconfigure(0, weight=1)
        win.columnconfigure(0, weight=1)
        paned, tree, right_frame = _build_spyder_sidebar(win)
        paned.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        grid_ctrl = _build_spyder_grid_panel(right_frame, model, on_close=win.destroy, open_popup_cb=open_popup)
        nodes = {}
        for label, section in sidebar.items():
            rows, cols = section.shape()
            node = tree.insert("", "end", text=label, values=("Table", f"({rows}, {cols})"))
            nodes[node] = section

        def on_tree_select(*_: Any) -> None:
            selected = tree.selection()
            if selected:
                section = nodes[selected[0]]
                grid_ctrl["load_model"](section, push_history=False, crumb_label=section.title)

        tree.bind("<<TreeviewSelect>>", on_tree_select)
        initial_node = next((node for node, section in nodes.items() if section is model), None)
        if initial_node is not None:
            tree.selection_set(initial_node)
            tree.focus(initial_node)

    if owns_root:
        win.mainloop()


def show_structured_data_explorer(
    data_dir: Path = DEFAULT_DATA_DIR,
    parent: Optional[Any] = None,
    output_stem: Optional[str] = None,
    observation_space_info: Optional[Dict[str, Tuple[str, str]]] = None,
    action_space_info: Optional[Dict[str, Tuple[str, str]]] = None,
) -> None:
    """Open a maximized Spyder-style Variable & DataFrame Explorer for Two-Tier NPZ data."""
    import tkinter as tk
    from tkinter import ttk

    owns_root = parent is None
    if owns_root:
        ensure_repo_root_cwd(REPO_ROOT)

    workspace = StructuredDatasetWorkspace(
        data_dir=data_dir,
        output_stem=output_stem,
        observation_space_info=observation_space_info,
        action_space_info=action_space_info,
    )
    feat_cols = workspace.feature_columns
    act_cols = workspace.action_columns
    splits_order = workspace.splits_order

    win = tk.Tk() if owns_root else tk.Toplevel(parent)
    win.title(f"Spyder Structured Data Explorer — {workspace.data_dir}")
    _maximize_tk_window(win)

    def open_popup_table(sub_model: GridTableModel) -> None:
        show_spyder_table(sub_model, title=sub_model.title, parent=win, maximized=False)

    outer = ttk.Frame(win, padding=8)
    outer.grid(row=0, column=0, sticky="nsew")
    win.rowconfigure(0, weight=1)
    win.columnconfigure(0, weight=1)
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(2, weight=1)

    banner = ttk.LabelFrame(
        outer,
        text=(
            f"Two-Tier Dataset — Observation Space ({len(feat_cols)} Features) "
            f"& Action Space ({len(act_cols)} Physical Labels)"
        ),
        padding=(10, 6),
    )
    banner.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    banner.columnconfigure(1, weight=1)

    split_summary_parts = []
    for sp in splits_order:
        if sp in workspace.splits_raw:
            n_sp = len(workspace.splits_raw[sp]["X"])
            n_tr = len(set(workspace.splits_raw[sp]["run_id"].tolist()))
            split_summary_parts.append(f"{sp}: {n_sp:,} samples ({n_tr} trials)")
    summary_text = (
        f"Observation Space X [{len(feat_cols)}]: {', '.join(feat_cols[:6])}, ...   |   "
        f"Action Space Y [{len(act_cols)}]: {', '.join(act_cols)}   |   "
        + "  •  ".join(split_summary_parts)
    )
    ttk.Label(banner, text=summary_text, foreground=CP1_LABEL_FG).grid(row=0, column=0, columnspan=2, sticky="w")

    quick_bar = ttk.Frame(outer)
    quick_bar.grid(row=1, column=0, sticky="ew", pady=(0, 8))

    ttk.Label(quick_bar, text="Active Split:", font=("Ubuntu", 10, "bold"), foreground=CP1_LABEL_FG).pack(
        side="left", padx=(0, 8)
    )
    split_var = tk.StringVar(master=win, value=workspace.default_split())
    split_combo = ttk.Combobox(
        quick_bar,
        textvariable=split_var,
        values=[s for s in splits_order if s in workspace.splits_raw],
        state="readonly",
        width=12,
    )
    split_combo.pack(side="left", padx=(0, 12))

    paned, tree, right_frame = _build_spyder_sidebar(outer)
    paned.grid(row=2, column=0, sticky="nsew")

    initial_table = workspace.build_split_table(workspace.default_split(), mode="obs_act")
    grid_ctrl = _build_spyder_grid_panel(
        right_frame,
        initial_table,
        on_close=win.destroy,
        open_popup_cb=open_popup_table,
    )

    node_factories: Dict[str, Callable[[], GridTableModel]] = {}

    meta_root_id = tree.insert("", "end", text="Dataset Metadata & Contract", values=("JSON", "root"), open=True)
    node_factories[meta_root_id] = workspace.build_meta_root_table

    total_dims = len(feat_cols) + len(act_cols)
    spec_id = tree.insert(meta_root_id, "end", text="Obs & Action Space Spec", values=("Contract", f"({total_dims},)"))
    node_factories[spec_id] = workspace.build_space_spec_table

    src_count = len(workspace.meta.get("sources", []))
    src_id = tree.insert(meta_root_id, "end", text="Trials Inventory (sources)", values=("Sub-Data", f"({src_count},)"))
    node_factories[src_id] = workspace.build_sources_table

    for m_key in ("frame_contract", "camera", "splits", "rejection_counts", "px4_alignment", "sidecar_arrays"):
        if m_key in workspace.meta:
            val_obj = workspace.meta[m_key]
            nid = tree.insert(meta_root_id, "end", text=m_key, values=(type(val_obj).__name__, f"{len(val_obj)}"))
            node_factories[nid] = lambda k=m_key, v=val_obj: GridTableModel.from_nested_object(f"meta.{k}", v)

    for sp in splits_order:
        if sp not in workspace.splits_raw:
            continue
        d = workspace.splits_raw[sp]
        n_sp = len(d["X"])
        sp_node = tree.insert(
            "",
            "end",
            text=f"{sp}.npz",
            values=("Two-Tier Split", f"({n_sp:,})"),
            open=(sp == workspace.default_split()),
        )
        node_factories[sp_node] = lambda s=sp: workspace.build_split_table(s, mode="obs_act")

        obs_id = tree.insert(sp_node, "end", text="Observation Space (X)", values=(f"{len(feat_cols)} Features", f"({n_sp}, {len(feat_cols)})"))
        node_factories[obs_id] = lambda s=sp: workspace.build_split_table(s, mode="obs")

        act_id = tree.insert(sp_node, "end", text="Action Space (Y)", values=(f"{len(act_cols)} Labels", f"({n_sp}, {len(act_cols)})"))
        node_factories[act_id] = lambda s=sp: workspace.build_split_table(s, mode="act")

        xy_id = tree.insert(sp_node, "end", text="Obs + Action (X + Y)", values=("Feature+Label", f"({n_sp}, {total_dims})"))
        node_factories[xy_id] = lambda s=sp: workspace.build_split_table(s, mode="obs_act")

        st_id = tree.insert(sp_node, "end", text="Distribution Stats", values=("Quantiles", f"({total_dims}, 8)"))
        node_factories[st_id] = lambda s=sp: workspace.build_split_stats_table(s)

        full_id = tree.insert(sp_node, "end", text="Full Two-Tier Table", values=("All Cols", f"({n_sp},)"))
        node_factories[full_id] = lambda s=sp: workspace.build_split_table(s, mode="full")

        side_node = tree.insert(sp_node, "end", text="Tier 2 PX4 Sidecars", values=("PX4 + Stats", f"({n_sp},)"))
        node_factories[side_node] = lambda s=sp: workspace.build_split_table(s, mode="sidecar")
        for sk in ("px4_rpy_rad", "px4_accel_ned_mps2", "px4_ang_vel_radps", "px4_traj_sp", "stats_pos_ned_m"):
            if sk in d:
                sh = str(d[sk].shape)
                sk_id = tree.insert(side_node, "end", text=sk, values=("Sidecar", sh))
                node_factories[sk_id] = lambda s=sp, key=sk: workspace.build_split_table(s, mode="sidecar", sidecar_key=key)

        if "runner_mode" in d and n_sp > 0:
            modes_node = tree.insert(sp_node, "end", text="Sub-Data: By Runner Mode", values=("Modes", "drill-down"))
            node_factories[modes_node] = lambda s=sp: workspace.build_split_table(s, mode="full")
            modes_arr = d["runner_mode"].astype(str)
            runs_arr = d["run_id"].astype(str)
            segs_arr = d["segment_id"]
            for m_name in sorted(set(modes_arr.tolist())):
                m_mask = modes_arr == m_name
                m_cnt = int(np.sum(m_mask))
                m_id = tree.insert(modes_node, "end", text=m_name, values=("Runner Mode", f"({m_cnt},)"))
                node_factories[m_id] = lambda s=sp, mn=m_name: workspace.build_split_table(s, mode="obs_act", runner_mode_filter=mn)

                for rid in sorted(set(runs_arr[m_mask].tolist())):
                    r_mask = m_mask & (runs_arr == rid)
                    r_cnt = int(np.sum(r_mask))
                    r_id = tree.insert(m_id, "end", text=rid, values=("Trial", f"({r_cnt},)"))
                    node_factories[r_id] = lambda s=sp, r=rid: workspace.build_split_table(s, mode="full", run_id_filter=r)
                    unique_segs = sorted(set(int(x) for x in segs_arr[r_mask].tolist()))
                    if len(unique_segs) > 1:
                        for seg in unique_segs:
                            s_cnt = int(np.sum(r_mask & (segs_arr == seg)))
                            sg_id = tree.insert(r_id, "end", text=f"segment #{seg}", values=("Segment", f"({s_cnt},)"))
                            node_factories[sg_id] = lambda s=sp, r=rid, sg=seg: workspace.build_split_table(
                                s, mode="full", run_id_filter=r, seg_filter=sg
                            )

    def on_tree_select(*_: Any) -> None:
        sel = tree.selection()
        if not sel:
            return
        nid = sel[0]
        factory = node_factories.get(nid)
        if factory is not None:
            new_m = factory()
            grid_ctrl["load_model"](new_m, push_history=False, crumb_label=new_m.title)

    tree.bind("<<TreeviewSelect>>", on_tree_select)

    def load_quick_mode(mode_key: str) -> None:
        sp = split_var.get()
        if sp not in workspace.splits_raw:
            return
        if mode_key == "stats":
            m = workspace.build_split_stats_table(sp)
        elif mode_key == "spec":
            m = workspace.build_space_spec_table()
        elif mode_key == "sources":
            m = workspace.build_sources_table()
        else:
            m = workspace.build_split_table(sp, mode=mode_key)
        grid_ctrl["load_model"](m, push_history=False, crumb_label=m.title)

    split_combo.bind("<<ComboboxSelected>>", lambda *_: load_quick_mode("obs_act"))

    quick_buttons = [
        (f"Observation Space (X: {len(feat_cols)})", "obs"),
        (f"Action Space (Y: {len(act_cols)})", "act"),
        (f"Obs + Action ({total_dims})", "obs_act"),
        ("PX4 Sidecars (Tier 2)", "sidecar"),
        ("Full Table", "full"),
        ("Space Spec & Stats", "spec"),
        ("Trials Inventory (Sub-Data)", "sources"),
    ]
    for btn_text, m_key in quick_buttons:
        make_luxury_button(
            quick_bar,
            text=btn_text,
            command=lambda mk=m_key: load_quick_mode(mk),
            padx=16,
            pady=8,
            font_size=10,
        ).pack(side="left", padx=5, pady=4)

    if owns_root:
        win.mainloop()
