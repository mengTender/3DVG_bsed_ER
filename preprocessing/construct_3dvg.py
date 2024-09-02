import math
import os

import utils
import torch
import numpy as np
import dgl


def cosine_similarity(vector_a, vector_b):

    dot_product = np.dot(vector_a, vector_b)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    cosine_sim = dot_product / (norm_a * norm_b)

    return cosine_sim


def edge_construct(agent_info, object_info, gaze_angle, depth_map):
    # unit vector
    vec_u = np.array([math.sin(gaze_angle[0]), math.cos(gaze_angle[1])])
    vec_v = np.array([0, math.cos(gaze_angle[0])])
    # direction vector
    vec_a_ga = np.array([math.sin(gaze_angle[0]), math.cos(gaze_angle[0]) * math.tan(gaze_angle[0]), math.cos(gaze_angle[1])])
    # center coordinate
    p_center_agent = utils.get_object_center(agent_info)
    p_center_obj = utils.get_object_center(object_info)
    p_dep_agent = depth_map(p_center_agent)
    p_dep_obj = depth_map(p_center_obj)
    z_obj = p_dep_obj - p_dep_agent
    # distance
    dis = math.sqrt((p_center_agent[0] - p_center_obj[0])^2 + (p_center_agent[1] - p_center_obj[1])^2)
    # direction vector
    vec_obj = np.array([p_center_obj[0] - p_center_agent[0], p_center_obj[1] - p_center_agent[1], z_obj])
    # weight
    cos_obj = cosine_similarity(vec_a_ga, vec_obj)
    weight_obj = cos_obj * (1 / dis)
    return weight_obj


def graph_construct(objects_info, agent_index, gaze_angle, depth_map):
    agent_info = objects_info[agent_index]
    weights = []
    for i in range(0, len(objects_info)):
        if i != agent_index:
            weight = edge_construct(agent_info, objects_info[i], gaze_angle, depth_map)
            weights.append(weight)
    for i in range(len(weights)):
        weights[i] = (sum(weights) - weights[i]) / sum(weights)
    return weights


def build_graph_with_npy_features(agent_index, node_feature_file_ls, edge_weights, min_nodes=5):
    num_nodes = len(node_feature_file_ls)
    edge_weights.insert(0, agent_index)
    node_features_ls = []
    for node_feature_file in node_feature_file_ls:
        node_feature = np.load(node_feature_file)
        if node_feature.ndim > 2:
            node_feature = node_feature.reshape(node_feature.shape[0], -1)
        node_features_ls.append(node_feature)

    if num_nodes < min_nodes:
        num_missing_nodes = min_nodes - num_nodes
        for i in range(0, num_missing_nodes):
            zero_feature = np.zeros((node_feature.shape[0], 1))
            node_features_ls.append(zero_feature)
        num_nodes = min_nodes
    node_features_tensor_ls = []
    for i in range(len(node_features_ls)):
        node_feature = torch.tensor(node_features_ls[i], dtype=torch.float32)
        node_features_tensor_ls.append(node_feature)

    # construct empty graph
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    # add edge
    for i in range(num_nodes):
        if i != agent_index:
            g.add_edges(agent_index, i)
            g.edges[agent_index, i].data['weight'] = torch.tensor([edge_weights[i]], dtype=torch.float32)

    g.ndata['feat'] = torch.tensor(node_features_tensor_ls)
    return g

