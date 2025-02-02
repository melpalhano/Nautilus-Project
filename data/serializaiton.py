import trimesh
import numpy as np
import networkx as nx
from .data_utils import discretize, undiscretize


def serialize_mesh(mesh: trimesh.Trimesh, special_token=-2):
    sequence = []
    unvisited = np.full(len(mesh.faces), True)
    degrees = mesh.vertex_degree.copy()

    random_next_center = True
    graph = mesh.vertex_adjacency_graph
    while sum(unvisited):
        unvisited_faces = mesh.faces[unvisited]

        def find_necessary_center(neighbors, graph, degrees, mesh, unvisited):
            for neighbor in neighbors:
                neighbor_neighbors = np.array(list(nx.neighbors(graph, neighbor)))
                one_degree_neighbors = neighbor_neighbors[degrees[neighbor_neighbors] == 1]

                for one_deg_nbr in one_degree_neighbors:
                    one_deg_nbr_faces = mesh.vertex_faces[one_deg_nbr]
                    if np.any(unvisited[one_deg_nbr_faces[one_deg_nbr_faces != -1]]):
                        return neighbor
                    else:
                        raise ValueError
            return None

        if not random_next_center:
            last_vertex = res[-1]
            last_neighbors = [nbr for nbr in nx.neighbors(graph, last_vertex) if degrees[nbr] > 0]

            if last_neighbors:
                max_deg_vertex = find_necessary_center(last_neighbors, graph, degrees, mesh, unvisited)

                if max_deg_vertex is None:
                    max_deg_vertex = last_neighbors[np.argmax(degrees[last_neighbors])]
                    if degrees[max_deg_vertex] < 5:
                        random_next_center = True

        if random_next_center:
            # select the patch center
            cur_face = unvisited_faces[0]
            max_deg_vertex_id = np.argmax(degrees[cur_face])
            max_deg_vertex = cur_face[max_deg_vertex_id]

        # find all connected faces
        selected_faces = []
        for face_idx in mesh.vertex_faces[max_deg_vertex]:
            if face_idx != -1 and unvisited[face_idx]:
                face = mesh.faces[face_idx]
                u, v = sorted([vertex for vertex in face if vertex != max_deg_vertex])
                selected_faces.append([u, v, face_idx])
                
        face_patch = set()
        selected_faces = sorted(selected_faces)
        
        # select the start vertex, select it if it only appears once (the start or end), 
        # else select the lowest index
        cnt = {}
        for u, v, _ in selected_faces:
            cnt[u] = cnt.get(u, 0) + 1
            cnt[v] = cnt.get(v, 0) + 1
        starts = []
        for vertex, num in cnt.items():
            if num == 1:
                starts.append(vertex)
        start_idx = min(starts) if len(starts) else selected_faces[0][0]
        
        res = [start_idx]
        while len(res) <= len(selected_faces):
            vertex = res[-1]
            for u_i, v_i, face_idx_i in selected_faces:
                if face_idx_i not in face_patch and vertex in (u_i, v_i):
                    u_i, v_i = (u_i, v_i) if vertex == u_i else (v_i, u_i)
                    res.append(v_i)
                    face_patch.add(face_idx_i)
                    break
            
            if res[-1] == vertex:
                break
        
        # reduce the degree of related vertices and mark the visited faces
        degrees[max_deg_vertex] = len(selected_faces) - len(res) + 1
        for pos_idx, vertex in enumerate(res):
            if pos_idx in [0, len(res) - 1]:
                degrees[vertex] -= 1
            else:
                degrees[vertex] -= 2
        for face_idx in face_patch:
            unvisited[face_idx] = False 
        sequence.extend([mesh.vertices[max_deg_vertex]] + [mesh.vertices[vertex_idx] for vertex_idx in res] + [[special_token] * 3])
        
    assert sum(degrees) == 0, 'All degrees should be zero'

    return np.array(sequence)


def coordinates_compression(sequence, u_size=1024, v_size=2048, special_token=-2, quant_bit=7):
    assert u_size % 2 == 0 and v_size % 2 == 0
    # prepare coordinates
    sp_mask = sequence != special_token
    sp_mask = np.all(sp_mask, axis=1)
    coords = sequence[sp_mask].reshape(-1, 3)
    coords = discretize(coords, num_discrete=2**quant_bit)

    # convert [x, y, z] to [u_id, v_id]
    num_discrete_coords = 2**quant_bit
    assert coords.max() < num_discrete_coords
    sum_coords = coords[:, 0] * num_discrete_coords**2 + coords[:, 1] * num_discrete_coords + coords[:, 2]
    u_id = sum_coords // v_size
    v_id = sum_coords % v_size
    assert np.all(u_id * v_size + v_id == sum_coords)
    v_id += u_size

    uv_coords = np.concatenate([u_id[..., None], v_id[..., None]], axis=-1).astype(np.int64)
    sequence[:, :2][sp_mask] = uv_coords
    sequence = sequence[:, :2]
    
    # convert to codes
    codes = []
    cur_uv_id = special_token
    for i in range(len(sequence)):
        if sequence[i, 0] == special_token:
            cur_uv_id = special_token
        elif sequence[i, 0] == cur_uv_id:
                codes.append(sequence[i, 1])
        else:
            if cur_uv_id == special_token:
                u_id = sequence[i, 0] + u_size + v_size
            else:
                u_id = sequence[i, 0]
            codes.extend([u_id, sequence[i, 1]])
            cur_uv_id = u_id

    codes = np.array(codes).astype(np.int64)
    sequence = codes
    
    return sequence.flatten()


def decode_shell(sequence, u_size=16, v_size=16, num_discrete_coors=2 ** 7):
    # decode from compressed representation
    res = []
    res_u = 0
    for token_id in range(len(sequence)):
        if u_size + v_size > sequence[token_id] >= u_size:
            res.append([res_u, sequence[token_id]])
        elif u_size > sequence[token_id] >= 0:
            res_u = sequence[token_id]
        else:
            print('[Warning] too large v idx!', token_id, sequence[token_id])
    sequence = np.array(res)
    
    u_id, v_id = np.array_split(sequence, 2, axis=-1)
    
    # from hash representation to xyz
    coords = []
    v_id -= u_size
    sum_coords = u_id * v_size + v_id
    for i in [2, 1, 0]:
        axis = (sum_coords // num_discrete_coors**i)
        sum_coords %= num_discrete_coors**i
        coords.append(axis)
    
    coords = np.concatenate(coords, axis=-1) # (nf 3)
    
    # back to continuous space
    coords = undiscretize(coords, num_discrete=num_discrete_coors)

    return coords


def detokenize(sequence, u_size=1024, v_size=2048, num_discrete_coors=2 ** 7):
    uv_size = u_size + v_size
    start_idx = 0
    vertices = []
    for i in range(len(sequence)):
        sub_seq = []
        if (uv_size <= sequence[i] < uv_size + u_size or i == len(sequence) - 1) and (i != 0):
            sub_seq = sequence[start_idx:i] if i != len(sequence) - 1 else sequence[start_idx: i+1]
            sub_seq[0] -= uv_size if uv_size <= sub_seq[0] < uv_size + u_size else 0
            sub_seq = decode_shell(sub_seq, u_size=u_size, v_size=v_size, num_discrete_coors=num_discrete_coors)
            start_idx = i
            
        if len(sub_seq):
            center, sub_seq = sub_seq[0], sub_seq[1:]
            for j in range(len(sub_seq) - 1):
                vertices.extend([center.reshape(1, 3), sub_seq[j].reshape(1, 3), sub_seq[j+1].reshape(1, 3)])
        
    return np.concatenate(vertices, axis=0)