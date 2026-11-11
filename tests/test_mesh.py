import json
import struct

import numpy as np
import pytest

from btseg.mesh import segmentation_glb, segmentation_surfaces, surface, write_glb


def ball(shape, center, radius):
    grid = np.indices(shape).transpose(1, 2, 3, 0)
    return np.linalg.norm(grid - np.asarray(center), axis=-1) <= radius


def signed_volume(s):
    v = s.vertices.astype(np.float64)[s.faces]
    return np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6


@pytest.mark.parametrize(
    "affine",
    [np.diag([1.0, 1.0, 1.0, 1.0]), np.diag([-1.2, -1.2, 2.0, 1.0])],  # RAS and LPS-like
)
def test_surface_is_closed_outward_and_true_to_scale(affine):
    mask = ball((40, 40, 30), (20, 20, 15), 8)
    s = surface(mask, affine, "ball", (1, 0, 0, 1))
    voxel_mm3 = abs(np.linalg.det(affine[:3, :3]))
    expected = mask.sum() * voxel_mm3
    # positive signed volume = consistent outward winding; close to the voxel volume
    assert signed_volume(s) == pytest.approx(expected, rel=0.15)
    center = s.vertices.mean(axis=0)
    outward = np.einsum("ij,ij->i", s.normals, s.vertices - center)
    assert (outward > 0).mean() > 0.95
    assert s.vertices.dtype == np.float32 and s.faces.dtype == np.uint32


def test_axes_face_the_patient():
    # a blob on the patient's right (+x in RAS) and superior (+z) of a centered one
    mask = np.zeros((40, 40, 40), bool)
    mask[30:34, 18:22, 30:34] = True
    s = surface(mask, np.eye(4), "blob", (1, 1, 1, 1), sigma=0)
    x, y, z = s.vertices.mean(axis=0)
    assert x < -25 and y > 25 and 15 < z < 25  # glTF: -x = patient right, +y = superior


def test_empty_and_tiny_regions():
    assert surface(np.zeros((8, 8, 8), bool), np.eye(4), "e", (1, 1, 1, 1)) is None
    tiny = np.zeros((16, 16, 16), bool)
    tiny[8, 8, 8] = True  # one voxel survives smoothing as a small blob
    assert surface(tiny, np.eye(4), "t", (1, 1, 1, 1)) is not None


def test_segmentation_layers_and_centering():
    labels = np.zeros((48, 48, 40), np.uint8)
    labels[ball(labels.shape, (24, 24, 20), 10)] = 2
    labels[ball(labels.shape, (24, 24, 20), 6)] = 4
    labels[ball(labels.shape, (24, 24, 20), 3)] = 1
    brain = ball(labels.shape, (24, 24, 20), 18)
    layers = segmentation_surfaces(labels, np.eye(4), 4, brain)
    assert [s.name for s in layers] == ["brain", "whole tumor (edema)", "enhancing tumor",
                                        "necrotic core"]
    lo, hi = layers[0].vertices.min(axis=0), layers[0].vertices.max(axis=0)
    np.testing.assert_allclose((lo + hi) / 2, 0, atol=1e-4)  # brain centered at the origin
    assert [s.rgba[3] for s in layers] == [0.12, 0.3, 0.4, 1.0]
    assert segmentation_surfaces(np.zeros((8, 8, 8)), np.eye(4)) == []


def read_glb(path):
    data = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    assert (magic, version, total) == (b"glTF", 2, len(data))
    js_len, js_type = struct.unpack_from("<I4s", data, 12)
    assert js_type == b"JSON" and js_len % 4 == 0
    gltf = json.loads(data[20:20 + js_len])
    bin_len, bin_type = struct.unpack_from("<I4s", data, 20 + js_len)
    assert bin_type == b"BIN\0" and bin_len % 4 == 0
    return gltf, data[28 + js_len:28 + js_len + bin_len]


def test_glb_round_trip(tmp_path):
    labels = np.zeros((32, 32, 32), np.uint8)
    labels[ball(labels.shape, (16, 16, 16), 7)] = 2
    labels[ball(labels.shape, (16, 16, 16), 3)] = 1
    path = segmentation_glb(labels, np.diag([1.0, 1.0, 1.5, 1.0]), tmp_path / "t.glb")
    gltf, blob = read_glb(path)
    assert [m["name"] for m in gltf["meshes"]] == ["whole tumor (edema)", "necrotic core"]
    assert gltf["buffers"][0]["byteLength"] <= len(blob)
    surfaces = segmentation_surfaces(labels, np.diag([1.0, 1.0, 1.5, 1.0]))
    for mesh, s in zip(gltf["meshes"], surfaces, strict=True):
        prim = mesh["primitives"][0]
        pos = gltf["accessors"][prim["attributes"]["POSITION"]]
        view = gltf["bufferViews"][pos["bufferView"]]
        stored = np.frombuffer(blob, np.float32, pos["count"] * 3, view["byteOffset"])
        np.testing.assert_array_equal(stored.reshape(-1, 3), s.vertices)
        idx = gltf["accessors"][prim["indices"]]
        assert idx["count"] == s.faces.size and idx["componentType"] == 5125
    assert gltf["materials"][0]["alphaMode"] == "BLEND"
    assert gltf["materials"][1]["alphaMode"] == "OPAQUE"
    assert segmentation_glb(np.zeros((4, 4, 4)), np.eye(4), tmp_path / "e.glb") is None
    assert write_glb([], tmp_path / "empty.glb").exists()
