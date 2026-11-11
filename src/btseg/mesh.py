"""3D surface meshes of a segmentation, saved as a binary glTF (``.glb``) for 3D viewers.

Each region becomes a smooth surface (Gaussian-smoothed mask -> marching cubes) in world
millimetres (the NIfTI affine), so shapes and sizes are true to scale:

- brain outline (nonzero voxels of the image), faint gray, for context;
- whole tumor (WT, edema shell), translucent green;
- enhancing tumor (ET), semi-transparent yellow;
- necrotic core (NCR), opaque red.

Colors match the 2D overlay (:data:`btseg.viz.COLORS`). The mesh is for display only;
volumes and metrics always come from the voxel masks.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

from btseg.data.labels import NCR_LABEL
from btseg.viz import COLORS


@dataclass
class Surface:
    name: str
    vertices: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (M, 3) uint32
    normals: np.ndarray  # (N, 3) float32, unit, pointing outward
    rgba: tuple[float, float, float, float]


def _ras_to_gltf(points: np.ndarray) -> np.ndarray:
    """RAS+ mm -> glTF axes (Y up, +Z toward the viewer): the viewer faces the patient."""
    return np.stack([-points[:, 0], points[:, 2], points[:, 1]], axis=1)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # area-weighted
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_n)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(length > 0, length, 1.0)


def surface(
    mask: np.ndarray,
    affine: np.ndarray,
    name: str,
    rgba: tuple[float, float, float, float],
    sigma: float = 1.0,
    step: int = 1,
) -> Surface | None:
    """Smooth closed surface of a binary mask in glTF-oriented world mm; None if empty."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    margin = int(np.ceil(3 * sigma)) + step + 1
    lo = np.maximum(np.argwhere(mask).min(axis=0) - margin, 0)
    hi = np.argwhere(mask).max(axis=0) + margin + 1
    crop = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    field = np.pad(gaussian_filter(crop, sigma) if sigma > 0 else crop, 1)  # closed at edges
    level = min(0.5, 0.5 * float(field.max()))  # keep tiny regions visible after smoothing
    verts, faces, _, _ = marching_cubes(field, level=level, step_size=step,
                                        allow_degenerate=False)
    if len(faces) == 0:
        return None
    ijk = verts + lo - 1  # undo the crop and the 1-voxel pad
    world = ijk @ affine[:3, :3].T + affine[:3, 3]
    vertices = _ras_to_gltf(world)
    faces = faces.astype(np.int64)
    tri = vertices[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    outward = np.einsum("ij,ij->", face_n, tri.mean(axis=1) - vertices.mean(axis=0))
    if outward < 0:  # winding depends on the affine's handedness: make normals point out
        faces = faces[:, ::-1]
    return Surface(
        name=name,
        vertices=vertices.astype(np.float32),
        faces=np.ascontiguousarray(faces, dtype=np.uint32),
        normals=_vertex_normals(vertices, faces).astype(np.float32),
        rgba=rgba,
    )


def segmentation_surfaces(
    labels: np.ndarray,
    affine: np.ndarray,
    et_label: int = 4,
    brain_mask: np.ndarray | None = None,
) -> list[Surface]:
    """Brain (optional), WT, ET and NCR surfaces, centered on the brain (or the tumor)."""
    labels = np.asarray(labels)
    layers = []
    if brain_mask is not None:
        layers.append(surface(brain_mask, affine, "brain", (0.75, 0.75, 0.78, 0.12),
                              sigma=1.5, step=3))
    layers += [
        surface(labels > 0, affine, "whole tumor (edema)", (*COLORS["ED"], 0.3)),
        surface(labels == et_label, affine, "enhancing tumor", (*COLORS["ET"], 0.4)),
        surface(labels == NCR_LABEL, affine, "necrotic core", (*COLORS["NCR"], 1.0)),
    ]
    layers = [s for s in layers if s is not None]
    if layers:
        ref = layers[0].vertices  # brain if present, otherwise the whole tumor
        center = (ref.min(axis=0) + ref.max(axis=0)) / 2
        for s in layers:
            s.vertices = (s.vertices - center).astype(np.float32)
    return layers


def write_glb(surfaces: list[Surface], path: str | Path) -> Path:
    """Write surfaces as one binary glTF 2.0 file (one node, mesh and material each)."""
    path = Path(path)
    blob = bytearray()
    gltf: dict = {
        "asset": {"version": "2.0", "generator": "btseg"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(surfaces)))}],
        "nodes": [], "meshes": [], "materials": [], "accessors": [], "bufferViews": [],
    }

    def add_view(data: np.ndarray, target: int) -> int:
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": len(blob),
                                    "byteLength": data.nbytes, "target": target})
        blob.extend(data.tobytes())
        return len(gltf["bufferViews"]) - 1

    def add_accessor(view: int, ctype: int, count: int, kind: str, **extra) -> int:
        gltf["accessors"].append({"bufferView": view, "componentType": ctype,
                                  "count": count, "type": kind, **extra})
        return len(gltf["accessors"]) - 1

    for i, s in enumerate(surfaces):
        pos = add_accessor(add_view(s.vertices, 34962), 5126, len(s.vertices), "VEC3",
                           min=s.vertices.min(axis=0).tolist(),
                           max=s.vertices.max(axis=0).tolist())
        nrm = add_accessor(add_view(s.normals, 34962), 5126, len(s.normals), "VEC3")
        idx = add_accessor(add_view(s.faces.reshape(-1), 34963), 5125, s.faces.size, "SCALAR")
        gltf["materials"].append({
            "name": s.name,
            "pbrMetallicRoughness": {"baseColorFactor": [float(c) for c in s.rgba],
                                     "metallicFactor": 0.0, "roughnessFactor": 0.7},
            "alphaMode": "BLEND" if s.rgba[3] < 1 else "OPAQUE",
            "doubleSided": s.rgba[3] == 1,  # see-through layers: front faces only
        })
        gltf["meshes"].append({"name": s.name, "primitives": [{
            "attributes": {"POSITION": pos, "NORMAL": nrm}, "indices": idx, "material": i}]})
        gltf["nodes"].append({"name": s.name, "mesh": i})
    gltf["buffers"] = [{"byteLength": len(blob)}]

    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    blob.extend(b"\0" * (-len(blob) % 4))
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(blob), b"BIN\0") + bytes(blob))
    return path


def segmentation_glb(
    labels: np.ndarray,
    affine: np.ndarray,
    path: str | Path,
    et_label: int = 4,
    brain_mask: np.ndarray | None = None,
) -> Path | None:
    """Mesh a label map (plus optional brain outline) into a ``.glb``; None if all empty."""
    surfaces = segmentation_surfaces(labels, affine, et_label, brain_mask)
    return write_glb(surfaces, path) if surfaces else None
