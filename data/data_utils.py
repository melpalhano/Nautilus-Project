"""Mesh data utilities."""
import random
import networkx as nx
import torch
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def to_mesh(vertices, faces, transpose=True, post_process=False):
    if transpose:
        vertices = vertices[:, [1, 2, 0]]
        
    if faces.min() == 1:
        faces = (np.array(faces) - 1).tolist()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    if post_process:
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.fix_normals()
    return mesh


def face_to_cycles(face):
    """Find cycles in face."""
    g = nx.Graph()
    for v in range(len(face) - 1):
        g.add_edge(face[v], face[v + 1])
    g.add_edge(face[-1], face[0])
    return list(nx.cycle_basis(g))


def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    vert_center = 0.5 * (vert_min + vert_max)
    # vert_center = np.mean(vertices, axis=0)
    return vertices - vert_center


def normalize_vertices_scale(vertices, scale=0.95):
    """Scale the vertices so that the long axis of the bounding box is one."""
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    extents = (vert_max - vert_min).max()
    return 2.0 * scale * vertices / (extents + 1e-6)


def quantize_process_mesh(vertices, faces, tris=None, quantization_bits=8):
    """Quantize vertices, remove resulting duplicates and reindex faces."""
    vertices = discretize(vertices, num_discrete=2**quantization_bits)
    vertices, inv = np.unique(vertices, axis=0, return_inverse=True)

    # Sort vertices by z then y then x.
    sort_inds = np.lexsort(vertices.T)
    
    vertices = vertices[sort_inds]
    # Re-index faces and tris to re-ordered vertices.
    faces = [np.argsort(sort_inds)[inv[f]] for f in faces]
    if tris is not None:
        tris = np.array([np.argsort(sort_inds)[inv[t]] for t in tris])

    # Merging duplicate vertices and re-indexing the faces causes some faces to
    # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
    # sub-faces.
    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f)
        for c in cliques:
            c_length = len(c)
            # Only append faces with more than two verts.
            if c_length > 2:
                d = np.argmin(f)
                # Cyclically permute faces just that first index is the smallest.
                sub_faces.append([f[(d + i) % c_length] for i in range(c_length)])
                
                # d = np.argmin(c)
                # # Cyclically permute faces just that first index is the smallest.
                # sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
    faces = sub_faces
    if tris is not None:
        tris = np.array([v for v in tris if len(set(v)) == len(v)])

    # Sort faces by lowest vertex indices. If two faces have the same lowest
    # index then sort by next lowest and so on.
    faces.sort(key=lambda f: tuple(sorted(f)))
    if tris is not None:
        tris = tris.tolist()
        tris.sort(key=lambda f: tuple(sorted(f)))
        tris = np.array(tris)

    # After removing degenerate faces some vertices are now unreferenced.
    # Remove these.
    num_verts = vertices.shape[0]
    vert_connected = np.equal(
        np.arange(num_verts)[:, None], np.hstack(faces)[None]
    ).any(axis=-1)
    vertices = vertices[vert_connected]

    # Re-index faces and tris to re-ordered vertices.
    vert_indices = np.arange(num_verts) - np.cumsum(1 - vert_connected.astype("int"))
    faces = [vert_indices[f].tolist() for f in faces]
    if tris is not None:
        tris = np.array([vert_indices[t].tolist() for t in tris])

    return vertices, faces, tris


def process_mesh(vertices, faces, quantization_bits=8, augment=True, augment_dict=None):
    """Process mesh vertices and faces."""

    # Transpose so that z-axis is vertical.
    vertices = vertices[:, [2, 0, 1]]

    # Translate the vertices so that bounding box is centered at zero.
    vertices = center_vertices(vertices)

    if augment:
        vertices = augment_mesh(vertices, **augment_dict)

    # Scale the vertices so that the long diagonal of the bounding box is equal
    # to one.
    vertices = normalize_vertices_scale(vertices)

    # Quantize and sort vertices, remove resulting duplicates, sort and reindex
    # faces.
    vertices, faces, _ = quantize_process_mesh(
        vertices, faces, quantization_bits=quantization_bits
    )
    vertices = undiscretize(vertices, num_discrete=2**quantization_bits)

    # Discard degenerate meshes without faces.
    return {
        "vertices": vertices,
        "faces": faces,
    }


def load_process_mesh(mesh_obj_path, quantization_bits=8, augment=False, augment_dict=None, transformation=None):
    """Load obj file and process."""
    # Load mesh
    # vertices, faces = read_obj(mesh_obj_path)
    # It doesn't matter to meet "RuntimeWarning: invalid value encountered in cast"
    mesh = trimesh.load(mesh_obj_path, force='mesh', process=False)
    if transformation is not None:
        mesh.apply_transform(transformation)
    return process_mesh(mesh.vertices, mesh.faces, quantization_bits, augment=augment, augment_dict=augment_dict)


def augment_mesh(vertices, scale_min=0.95, scale_max=1.05, rotation=0., jitter_strength=0.):
    '''scale vertices by a factor in [0.75, 1.25]'''
    
    # vertices [nv, 3]
    for i in range(3):
        # Generate a random scale factor
        scale = random.uniform(scale_min, scale_max)    

        # independently applied scaling across each axis of vertices
        vertices[:, i] *= scale
    
    if rotation != 0.:
        axis = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
        radian = np.pi / 180 * rotation
        rotation = Rotation.from_rotvec(radian * np.array(axis))
        vertices =rotation.apply(vertices)
        
        
    if jitter_strength != 0.:
        jitter_amount = np.random.uniform(-jitter_strength, jitter_strength)
        vertices += jitter_amount
    
        
    return vertices


def jitter_vertices(vertices, jitter_strength=0.01):

    if jitter_strength != 0.:
        jitter_amount = np.random.uniform(-jitter_strength, jitter_strength, size=vertices.shape)
        vertices += jitter_amount

    return vertices


def discretize(
    t,
    continuous_range = (-1, 1),
    num_discrete: int = 128
):
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().astype(np.int32).clip(min = 0, max = num_discrete - 1)


def undiscretize(
    t,
    continuous_range = (-1, 1),
    num_discrete: int = 128
):
    lo, hi = continuous_range
    assert hi > lo

    try:
        t = t.astype(np.float32)
    except:
        t = t.to(torch.float32)

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo