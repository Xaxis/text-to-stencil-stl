#!/usr/bin/env python3
"""
stencil_stl.py (upgraded)

Creates 3D-printable stencil plates from text or arranged text items.

New features:
- --stencilify: creates stencil-style breaks for non-stencil fonts by subtracting periodic bars.
- More preview outputs: --debug-svg (final plate), --debug-cutout-svg (raw cutout), --debug-text-svg (text geometry).
- Fine control of stencil breaks: gap, period, angle, phase, direction.

Dependencies:
  pip install numpy shapely matplotlib trimesh mapbox_earcut
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from shapely.affinity import rotate as shp_rotate
from shapely.affinity import scale as shp_scale
from shapely.affinity import translate as shp_translate
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

MM_PER_PT = 25.4 / 72.0  # Matplotlib TextPath uses points


# ----------------------------
# Geometry helpers
# ----------------------------

def _ring_area(coords: Sequence[Tuple[float, float]]) -> float:
    if len(coords) < 3:
        return 0.0
    pts = np.asarray(coords, dtype=np.float64)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ensure_ring_orientation(coords: Sequence[Tuple[float, float]], ccw: bool) -> List[Tuple[float, float]]:
    pts = list(coords)
    if len(pts) < 3:
        return pts
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    area = _ring_area(pts)
    is_ccw = area > 0
    if is_ccw != ccw:
        pts = list(reversed(pts))
    return pts


def _fix_valid(geom):
    try:
        return geom.buffer(0)
    except Exception:
        return geom


def _as_polygons(geom) -> List[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    try:
        geoms = list(geom.geoms)
        return [g for g in geoms if isinstance(g, Polygon)]
    except Exception:
        return []


# ----------------------------
# Text to 2D polygons
# ----------------------------

def _polygons_from_textpath(tp: TextPath):
    loops = tp.to_polygons()
    polys: List[Polygon] = []
    for arr in loops:
        if len(arr) < 3:
            continue
        coords = [(float(x), float(y)) for x, y in arr]
        if coords[0] == coords[-1]:
            coords = coords[:-1]
        p = Polygon(coords)
        if not p.is_valid:
            p = _fix_valid(p)
        if p.area > 1e-6:
            polys.append(p)

    if not polys:
        return MultiPolygon([])

    # Determine containment depth for even-odd fill
    areas = [p.area for p in polys]
    order = sorted(range(len(polys)), key=lambda i: areas[i])  # small to large
    parent = [-1] * len(polys)

    for i in order:
        pi = polys[i]
        best = -1
        best_area = float("inf")
        for j in range(len(polys)):
            if i == j:
                continue
            pj = polys[j]
            if pj.area <= pi.area:
                continue
            if pj.contains(pi.representative_point()) and pj.contains(pi):
                if pj.area < best_area:
                    best_area = pj.area
                    best = j
        parent[i] = best

    depth = [0] * len(polys)
    for i in range(len(polys)):
        d = 0
        k = parent[i]
        while k != -1:
            d += 1
            k = parent[k]
            if d > 64:
                break
        depth[i] = d

    filled = [polys[i] for i in range(len(polys)) if depth[i] % 2 == 0]
    holes = [polys[i] for i in range(len(polys)) if depth[i] % 2 == 1]

    g = unary_union(filled)
    if holes:
        g = g.difference(unary_union(holes))
    return _fix_valid(g)


def text_geometry(text: str, font_path: Optional[str], font_size_mm: float):
    if not text.strip():
        return MultiPolygon([])
    size_pt = font_size_mm / MM_PER_PT
    fp = FontProperties(fname=font_path) if font_path else FontProperties()
    tp = TextPath((0, 0), text, size=size_pt, prop=fp, usetex=False)
    g = _polygons_from_textpath(tp)
    return _fix_valid(shp_scale(g, xfact=MM_PER_PT, yfact=MM_PER_PT, origin=(0, 0)))


# ----------------------------
# Layout
# ----------------------------

@dataclass
class LayoutLine:
    text: str
    size_mm: float
    x: float = 0.0
    y: float = 0.0
    rotate_deg: float = 0.0
    align: str = "center"  # left, center, right


def layout_multiline(text: str, font_path: Optional[str], font_size_mm: float,
                     line_spacing: float, align: str):
    lines = text.splitlines()
    if not lines:
        return MultiPolygon([])

    geoms = []
    bnds = []
    for ln in lines:
        g = text_geometry(ln, font_path, font_size_mm)
        geoms.append(g)
        bnds.append(g.bounds if not g.is_empty else (0, 0, 0, 0))

    line_height = font_size_mm * line_spacing
    widths = [(b[2] - b[0]) for b in bnds]
    block_w = max(widths) if widths else 0.0

    total_h = line_height * (len(lines) - 1)
    y0 = total_h / 2.0

    placed = []
    for i, (g, b) in enumerate(zip(geoms, bnds)):
        if g.is_empty:
            continue
        line_w = b[2] - b[0]
        if align == "left":
            dx = -block_w / 2.0
        elif align == "right":
            dx = block_w / 2.0 - line_w
        else:
            dx = -line_w / 2.0

        g2 = shp_translate(g, xoff=-b[0], yoff=0)  # normalize minx to 0
        g2 = shp_translate(g2, xoff=dx, yoff=y0 - i * line_height)
        placed.append(g2)

    return _fix_valid(unary_union(placed)) if placed else MultiPolygon([])


def layout_from_json(layout_path: str, font_path: Optional[str], default_size_mm: float):
    data = json.loads(Path(layout_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Layout JSON must be a list of text items.")

    geoms = []
    for item in data:
        if not isinstance(item, dict) or "text" not in item:
            continue
        ln = LayoutLine(
            text=str(item["text"]),
            size_mm=float(item.get("size_mm", default_size_mm)),
            x=float(item.get("x", 0.0)),
            y=float(item.get("y", 0.0)),
            rotate_deg=float(item.get("rotate_deg", 0.0)),
            align=str(item.get("align", "center")).lower(),
        )

        g = text_geometry(ln.text, font_path, ln.size_mm)
        if g.is_empty:
            continue

        minx, miny, maxx, maxy = g.bounds
        if ln.align == "left":
            g = shp_translate(g, xoff=-minx, yoff=0)
        elif ln.align == "right":
            g = shp_translate(g, xoff=-maxx, yoff=0)
        else:
            g = shp_translate(g, xoff=-(minx + maxx) / 2.0, yoff=0)

        if abs(ln.rotate_deg) > 1e-6:
            g = shp_rotate(g, ln.rotate_deg, origin=(0, 0), use_radians=False)

        g = shp_translate(g, xoff=ln.x, yoff=ln.y)
        geoms.append(g)

    return _fix_valid(unary_union(geoms)) if geoms else MultiPolygon([])


# ----------------------------
# Bridges (counter islands)
# ----------------------------

def _bridge_holes_in_cutout(cutout: Polygon, bridge_width: float) -> Polygon:
    if bridge_width <= 0 or len(cutout.interiors) == 0:
        return cutout

    poly = cutout
    outer = Polygon(poly.exterior.coords)
    outer_minx, _, outer_maxx, _ = outer.bounds

    for ring in list(poly.interiors):
        coords = list(ring.coords)
        if coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) < 3:
            continue

        hx, hy = min(coords, key=lambda p: p[0])

        far_left = outer_minx - (outer_maxx - outer_minx) * 2.0 - 200.0
        ray = LineString([(hx, hy), (far_left, hy)])
        inter = ray.intersection(LineString(poly.exterior.coords))

        if inter.is_empty:
            far_right = outer_maxx + (outer_maxx - outer_minx) * 2.0 + 200.0
            ray = LineString([(hx, hy), (far_right, hy)])
            inter = ray.intersection(LineString(poly.exterior.coords))
            if inter.is_empty:
                continue

        pts = []
        if inter.geom_type == "Point":
            pts = [inter]
        elif inter.geom_type == "MultiPoint":
            pts = list(inter.geoms)
        else:
            continue
        if not pts:
            continue

        if ray.coords[-1][0] < hx:
            cand = [p for p in pts if p.x < hx - 1e-6]
            if not cand:
                continue
            p_hit = max(cand, key=lambda p: p.x)
        else:
            cand = [p for p in pts if p.x > hx + 1e-6]
            if not cand:
                continue
            p_hit = min(cand, key=lambda p: p.x)

        x0, x1 = min(hx, p_hit.x), max(hx, p_hit.x)
        rect = Polygon(
            [(x0, hy - bridge_width / 2),
             (x1, hy - bridge_width / 2),
             (x1, hy + bridge_width / 2),
             (x0, hy + bridge_width / 2)]
        )

        bridge = rect.intersection(outer)
        if bridge.is_empty:
            continue

        poly = _fix_valid(poly.difference(bridge))
        if poly.is_empty:
            break

    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area, default=Polygon())
    return poly


def apply_bridges(cutout_geom, bridge_width: float):
    if bridge_width <= 0:
        return cutout_geom
    if isinstance(cutout_geom, Polygon):
        return _bridge_holes_in_cutout(cutout_geom, bridge_width)
    if isinstance(cutout_geom, MultiPolygon):
        bridged = [_bridge_holes_in_cutout(p, bridge_width) for p in cutout_geom.geoms]
        return _fix_valid(unary_union(bridged))
    return cutout_geom


# ----------------------------
# Stencilify: add breaks like a stencil font
# ----------------------------

def _bar_field(bounds, gap: float, period: float, phase: float, thickness: float):
    """
    Create a set of infinite-ish bars covering `bounds`.
    - Bars are axis-aligned rectangles before rotation.
    - gap: the empty part in each period you *remove* from text (i.e., break width)
    - thickness: bar thickness = gap (or user provided) - we treat bars as "cutters"
    """
    minx, miny, maxx, maxy = bounds
    w = maxx - minx
    h = maxy - miny
    pad = max(w, h) * 0.25 + period * 2 + 50  # generous
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad

    if period <= 0:
        period = max(10.0, (maxx - minx) / 10.0)
    if gap <= 0:
        gap = min(2.0, period * 0.25)

    # Bars will be vertical strips spaced by `period`
    bars = []
    x = minx + phase
    # Start far left so we cover everything
    while x < maxx + period:
        # One bar centered at x; thickness controls how much is removed
        bars.append(box(x - thickness / 2, miny, x + thickness / 2, maxy))
        x += period

    return unary_union(bars) if bars else MultiPolygon([])


def stencilify_cutout(cutout, gap: float, period: float, angle_deg: float, phase: float, direction: str):
    """
    Subtract a periodic bar field from the cutout, turning normal fonts into stencil-like breaks.
    direction:
      - "vertical": bars are vertical before rotation
      - "horizontal": bars are horizontal before rotation
    angle_deg rotates the bar field around the cutout center.
    """
    if cutout.is_empty or gap <= 0:
        return cutout

    minx, miny, maxx, maxy = cutout.bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    # Build bars in local axes; if horizontal, just swap by rotating 90 degrees first
    bounds = cutout.bounds
    thickness = gap

    bars = _bar_field(bounds, gap=gap, period=period, phase=phase, thickness=thickness)

    if direction == "horizontal":
        bars = shp_rotate(bars, 90.0, origin=(cx, cy), use_radians=False)

    if abs(angle_deg) > 1e-6:
        bars = shp_rotate(bars, angle_deg, origin=(cx, cy), use_radians=False)

    return _fix_valid(cutout.difference(bars))


# ----------------------------
# Triangulation + extrusion
# ----------------------------

class TriangulationError(RuntimeError):
    pass


def _try_import_triangulator():
    try:
        import mapbox_earcut as earcut  # type: ignore
        return "earcut", earcut
    except Exception:
        return None, None


def _polygon_to_earcut_inputs(poly: Polygon):
    exterior = _ensure_ring_orientation(list(poly.exterior.coords), ccw=True)
    rings = [exterior]
    rings_ccw = [True]
    for interior in poly.interiors:
        ring = _ensure_ring_orientation(list(interior.coords), ccw=False)
        if len(ring) >= 3:
            rings.append(ring)
            rings_ccw.append(False)

    verts: List[Tuple[float, float]] = []
    ring_end_indices: List[int] = []
    rings_meta: List[Tuple[int, int, bool]] = []

    for idx, ring in enumerate(rings):
        start = len(verts)
        verts.extend([(float(x), float(y)) for x, y in ring])
        ring_end_indices.append(len(verts))  # End index of this ring
        rings_meta.append((start, len(ring), rings_ccw[idx]))

    # Convert to numpy arrays with correct shapes for mapbox_earcut
    coords = np.asarray(verts, dtype=np.float64)  # Shape: (N, 2)
    ring_ends = np.asarray(ring_end_indices, dtype=np.uint32)  # Shape: (num_rings,)
    return coords, ring_ends, rings_meta


def _triangulate_polygon(poly: Polygon):
    backend, mod = _try_import_triangulator()
    if backend != "earcut":
        raise TriangulationError(
            "No triangulation backend found. Install:\n"
            "  pip install mapbox_earcut\n"
        )

    coords, ring_ends, rings_meta = _polygon_to_earcut_inputs(poly)

    # Call mapbox_earcut with correct arguments
    # coords: (N, 2) array of vertices
    # ring_ends: array of end indices for each ring
    if hasattr(mod, "triangulate_float64"):
        tri = mod.triangulate_float64(coords, ring_ends)
    elif hasattr(mod, "triangulate"):
        tri = mod.triangulate(coords, ring_ends)
    else:
        raise TriangulationError("mapbox_earcut installed but API not recognized.")

    tri = np.asarray(tri, dtype=np.int64).reshape(-1, 3)
    return tri, rings_meta, coords


def _build_extruded_mesh_from_polygon(poly: Polygon, height: float) -> trimesh.Trimesh:
    """
    Extrude a Shapely polygon to create a 3D mesh using trimesh's built-in extrusion.
    """
    poly = _fix_valid(poly)
    if poly.is_empty or poly.area < 1e-6:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))

    try:
        # Use trimesh's built-in polygon extrusion
        mesh = trimesh.creation.extrude_polygon(poly, height=height)
        return mesh
    except Exception as e:
        print(f"Warning: trimesh extrusion failed ({e}), falling back to manual extrusion")
        # Fallback to manual extrusion if trimesh fails
        return _manual_extrude_polygon(poly, height)


def _manual_extrude_polygon(poly: Polygon, height: float) -> trimesh.Trimesh:
    """
    Manually extrude a polygon (fallback method).
    """
    faces2d, rings_meta, verts2d = _triangulate_polygon(poly)
    n = verts2d.shape[0]

    def tri_signed_area(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    faces2d_oriented = faces2d.copy()
    for i in range(faces2d_oriented.shape[0]):
        ia, ib, ic = faces2d_oriented[i]
        a, b, c = verts2d[ia], verts2d[ib], verts2d[ic]
        if tri_signed_area(a, b, c) < 0:
            faces2d_oriented[i] = [ia, ic, ib]

    bottom = np.column_stack([verts2d, np.zeros((n, 1), dtype=np.float64)])
    top = np.column_stack([verts2d, np.full((n, 1), float(height), dtype=np.float64)])
    vertices = np.vstack([bottom, top])

    top_faces = faces2d_oriented + n
    bottom_faces = faces2d_oriented[:, ::-1]

    side_faces = []
    for start, length, ring_ccw in rings_meta:
        for k in range(length):
            i0 = start + k
            i1 = start + ((k + 1) % length)
            if ring_ccw:
                side_faces.append([i0, i1, i1 + n])
                side_faces.append([i0, i1 + n, i0 + n])
            else:
                side_faces.append([i0, i1, i1 + n])
                side_faces.append([i0, i1 + n, i0 + n])

    faces_all = np.vstack([bottom_faces, top_faces, np.asarray(side_faces, dtype=np.int64)])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces_all, process=True, validate=True)
    mesh.rezero()
    return mesh


def extrude_geometry(geom, height: float) -> trimesh.Trimesh:
    geom = _fix_valid(geom)
    if geom.is_empty:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))

    meshes = []
    for p in _as_polygons(geom):
        meshes.append(_build_extruded_mesh_from_polygon(p, height))

    if not meshes:
        return trimesh.Trimesh()

    # Concatenate and clean up the combined mesh
    combined = trimesh.util.concatenate(meshes)
    combined.merge_vertices()
    combined.remove_unreferenced_vertices()
    combined.fix_normals()
    combined.fill_holes()
    return combined


# ----------------------------
# Plate generation + SVG preview
# ----------------------------

def make_plate_with_cutouts(cutout2d, margin: float, corner_radius: float = 0.0):
    minx, miny, maxx, maxy = cutout2d.bounds
    minx -= margin
    miny -= margin
    maxx += margin
    maxy += margin

    if corner_radius > 0:
        base = box(minx, miny, maxx, maxy)
        plate = base.buffer(corner_radius).buffer(-corner_radius)
    else:
        plate = box(minx, miny, maxx, maxy)

    plate = _fix_valid(plate)
    cutout2d = _fix_valid(cutout2d)
    return _fix_valid(plate.difference(cutout2d))


def normalize_to_origin(geom):
    if geom.is_empty:
        return geom
    minx, miny, _, _ = geom.bounds
    return shp_translate(geom, xoff=-minx, yoff=-miny)


def write_svg(geom, path: str):
    if geom.is_empty:
        Path(path).write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>")
        return
    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    h = maxy - miny

    def ring_to_path(coords):
        coords = list(coords)
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        d = f"M {coords[0][0]-minx:.3f} {maxy-coords[0][1]:.3f} "
        for x, y in coords[1:]:
            d += f"L {x-minx:.3f} {maxy-y:.3f} "
        d += "Z "
        return d

    paths = []
    for p in _as_polygons(geom):
        d = ring_to_path(p.exterior.coords)
        for hole in p.interiors:
            d += ring_to_path(hole.coords)
        paths.append(d)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w:.3f}mm" height="{h:.3f}mm" viewBox="0 0 {w:.3f} {h:.3f}">
  <path d="{' '.join(paths)}" fill="black" fill-rule="evenodd" />
</svg>
"""
    Path(path).write_text(svg, encoding="utf-8")


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Convert text/arrangements into a 3D printable stencil STL.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Text for the stencil. Use \\n for multiple lines.")
    group.add_argument("--layout", type=str, help="Path to JSON layout file (advanced arrangements).")

    p.add_argument("--font", type=str, default=None, help="Path to TTF/OTF font file (recommended).")
    p.add_argument("--font-size", type=float, default=40.0, help="Font size in millimeters.")
    p.add_argument("--line-spacing", type=float, default=1.15, help="Line spacing multiplier.")
    p.add_argument("--align", type=str, default="center", choices=["left", "center", "right"])
    p.add_argument("--rotate", type=float, default=0.0, help="Rotate entire text block (degrees).")
    p.add_argument("--scale-to-width", type=float, default=0.0, help="Scale layout to fit this width (mm).")

    # Bridges
    p.add_argument("--bridge-width", type=float, default=0.0,
                   help="Add bridges for counter holes (mm). Usually 0 when using stencil fonts.")

    # Stencilify (break pattern)
    p.add_argument("--stencilify", action="store_true",
                   help="Turn any font into stencil-ish by cutting periodic breaks through the cutout.")
    p.add_argument("--stencil-gap", type=float, default=2.0,
                   help="Break width (mm) used by --stencilify.")
    p.add_argument("--stencil-period", type=float, default=16.0,
                   help="Spacing between breaks (mm) used by --stencilify.")
    p.add_argument("--stencil-angle", type=float, default=0.0,
                   help="Rotate break pattern (deg). 0 means vertical bars.")
    p.add_argument("--stencil-phase", type=float, default=0.0,
                   help="Shift break pattern along its axis (mm).")
    p.add_argument("--stencil-direction", type=str, default="vertical",
                   choices=["vertical", "horizontal"],
                   help="Break pattern base direction before rotation.")

    # Stencil mode
    p.add_argument("--mode", type=str, default="positive", choices=["positive", "negative"],
                   help="Stencil mode: 'positive' = text is solid (default), 'negative' = plate with text cut out.")

    # Plate (only used in negative mode)
    p.add_argument("--margin", type=float, default=12.0, help="Margin around text for plate (mm). Only used in negative mode.")
    p.add_argument("--corner-radius", type=float, default=0.0, help="Rounded corners radius (mm). Only used in negative mode.")
    p.add_argument("--plate-thickness", type=float, default=2.2, help="Plate thickness (mm).")

    # Output / debug
    p.add_argument("--out", type=str, required=True, help="Output STL path.")
    p.add_argument("--debug-text-svg", type=str, default=None, help="Write SVG of raw text geometry.")
    p.add_argument("--debug-cutout-svg", type=str, default=None, help="Write SVG of final cutout geometry.")
    p.add_argument("--debug-svg", type=str, default=None, help="Write SVG of final plate geometry.")
    return p.parse_args()


def main():
    args = parse_args()

    # Use default stencil font if none specified
    font_path = args.font
    if font_path is None:
        # Try to use the Warband Stencil font from the fonts directory
        script_dir = Path(__file__).parent
        default_font = script_dir / "fonts" / "Warband Stencil.otf"
        if default_font.exists():
            font_path = str(default_font)
            print(f"Using default stencil font: {default_font.name}")
        else:
            print("Warning: No font specified and default stencil font not found. Using system default.")
    elif not Path(font_path).exists():
        raise SystemExit(f"Font file not found: {font_path}")

    # 1) Text geometry (filled glyphs)
    if args.layout:
        text_geom = layout_from_json(args.layout, font_path, args.font_size)
    else:
        text_geom = layout_multiline(args.text, font_path, args.font_size, args.line_spacing, args.align)

    if text_geom.is_empty:
        raise SystemExit("No geometry produced from text. Check text/font parameters.")

    if args.debug_text_svg:
        write_svg(normalize_to_origin(text_geom), args.debug_text_svg)
        print(f"Wrote text SVG: {args.debug_text_svg}")

    # 2) Apply overall rotate
    if abs(args.rotate) > 1e-6:
        text_geom = shp_rotate(text_geom, args.rotate, origin=(0, 0), use_radians=False)

    # 3) Optional scale-to-width
    if args.scale_to_width and args.scale_to_width > 0:
        minx, _, maxx, _ = text_geom.bounds
        w = maxx - minx
        if w > 1e-6:
            s = args.scale_to_width / w
            text_geom = shp_scale(text_geom, xfact=s, yfact=s, origin=(0, 0))

    # 4) Cutout starts as the text shape (we subtract this from plate)
    cutout = _fix_valid(text_geom)

    # 5) Optional: stencilify (create breaks that keep islands attached)
    if args.stencilify:
        cutout = stencilify_cutout(
            cutout,
            gap=args.stencil_gap,
            period=args.stencil_period,
            angle_deg=args.stencil_angle,
            phase=args.stencil_phase,
            direction=args.stencil_direction,
        )

    # 6) Optional: bridges just for counters (typically used instead of stencilify)
    if args.bridge_width and args.bridge_width > 0:
        cutout = apply_bridges(cutout, args.bridge_width)

    if args.debug_cutout_svg:
        write_svg(normalize_to_origin(cutout), args.debug_cutout_svg)
        print(f"Wrote cutout SVG: {args.debug_cutout_svg}")

    # 7) Plate outline minus cutouts
    plate2d = make_plate_with_cutouts(cutout, margin=args.margin, corner_radius=args.corner_radius)
    plate2d = normalize_to_origin(plate2d)

    if args.debug_svg:
        write_svg(plate2d, args.debug_svg)
        print(f"Wrote plate SVG: {args.debug_svg}")

    # 8) Extrude & export
    mesh = extrude_geometry(plate2d, height=args.plate_thickness)
    if mesh.is_empty or len(mesh.faces) == 0:
        raise SystemExit("Mesh generation failed (empty mesh).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path.as_posix())

    print(f"Wrote STL: {out_path.resolve()}")
    print(f"Extents (mm): {mesh.extents}")


if __name__ == "__main__":
    main()
