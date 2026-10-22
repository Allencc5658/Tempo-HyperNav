import re
import json
import networkx as nx
from networkx.algorithms import isomorphism
import numpy as np
from scipy.spatial.distance import cosine
from heapq import heappush, heappop
from grakel import Graph, kernels

#计算场景图和目标图的匹配程度
class GraphMatcher:
    def __init__(self, graph1, graph2, llm=None):
        #graph1 场景图（有位置信息） 
        #graph2 目标图（无位置信息）
        self.graph1 = graph1
        self.graph2 = graph2
        self.llm = llm

        G1 = nx.DiGraph()
        for node in graph1['nodes']:
            G1.add_node(node['id'], position=node['position'])
        for edge in graph1['edges']:
            G1.add_edge(edge['source'], edge['target'], type=edge['type'])

        G2 = nx.DiGraph()
        for node in graph2['nodes']:
            G2.add_node(node['id'])  # No position assigned initially
        for edge in graph2['edges']:
            G2.add_edge(edge['source'], edge['target'], type=edge['type'])
        self.G1 = G1
        self.G2 = G2

    @staticmethod
    def _base_name(node_id):
        """从 'chair_0' 提取 'chair'，从 'potted plant' 保持原样"""
        return re.sub(r'_\d+$', '', node_id).lower().strip()

    def node_similarity(self, G1, G2):
        matched_pairs = self._semantic_match_nodes(G1, G2)
        num_matched = len(matched_pairs)

        if len(G2.nodes) == 0:
            node_overlap_ratio = 0.0
        else:
            node_overlap_ratio = num_matched / len(G2.nodes)

        degree_sim = 0.0
        if matched_pairs:
            for g1_node, g2_node in matched_pairs:
                degree_G1 = G1.degree(g1_node)
                degree_G2 = G2.degree(g2_node)
                degree_diff = abs(degree_G1 - degree_G2)
                max_degree = max(degree_G1, degree_G2)
                if max_degree > 0:
                    degree_sim += 1.0 - (degree_diff / max_degree)
            degree_sim /= len(matched_pairs)

        node_sim = (node_overlap_ratio + degree_sim) / 2.0
        return node_sim

    def edge_similarity(self, G1, G2):
        if G2.number_of_edges() == 0:
            return 1.0
        matched = 0
        for u2, v2, data2 in G2.edges(data=True):
            base_u2 = self._base_name(u2)
            base_v2 = self._base_name(v2)
            rel2 = data2.get('type', '')
            for u1, v1, data1 in G1.edges(data=True):
                if self._base_name(u1) == base_u2 and self._base_name(v1) == base_v2:
                    if data1.get('type', '') == rel2:
                        matched += 1
                        break
        return matched / G2.number_of_edges()

    def graph_edit_distance_heuristic(self, G1, G2):
        node_diff = abs(len(G1) - len(G2))
        edge_diff = abs(G1.number_of_edges() - G2.number_of_edges())
        return node_diff + edge_diff

    def apply_operation(self, operation, G, **kwargs):
        if operation == 'add_node':
            G.add_node(kwargs['node'])
        elif operation == 'remove_node':
            G.remove_node(kwargs['node'])
        elif operation == 'add_edge':
            G.add_edge(*kwargs['edge'])
        elif operation == 'remove_edge':
            G.remove_edge(*kwargs['edge'])

    def overlap(self):
        G1 = self.G1
        G2 = self.G2

        self.common_nodes = self.find_common_nodes(G1, G2)
        node_sim = self.node_similarity(G1, G2)
        edge_sim = self.edge_similarity(G1, G2)

        combined_sim = (node_sim + edge_sim) / 2
        return combined_sim

    def _semantic_match_nodes(self, G1, G2):
        """基于 base name 语义匹配，返回 [(g1_id, g2_id), ...] 配对列表"""
        g1_by_base = {}
        for n in G1.nodes:
            base = self._base_name(n)
            g1_by_base.setdefault(base, []).append(n)
        matched = []
        used_g1 = set()
        for n2 in G2.nodes:
            base2 = self._base_name(n2)
            candidates = g1_by_base.get(base2, [])
            for c in candidates:
                if c not in used_g1:
                    matched.append((c, n2))
                    used_g1.add(c)
                    break
        return matched

    def find_common_nodes(self, G1, G2):
        """返回匹配到的 G1 节点集合（保持下游 explore_remaining 兼容）"""
        matched_pairs = self._semantic_match_nodes(G1, G2)
        return set(g1_id for g1_id, _ in matched_pairs)

    def calculate_relative_positions(self, graph, common_nodes):
        """ Calculate relative positions of nodes within a graph using LLM for unknown positions """
        relative_positions = {}
        if not common_nodes:
            return relative_positions

        ref_node = list(common_nodes)[0]

        for node in common_nodes:
            if node == ref_node:
                continue
            prompt = f"Given the following information: {ref_node} and {node}. Please provide the relative position of {node} with respect to {ref_node} in the format [x, y]."
            response = self.llm(prompt)
            try:
                rel_pos_str = response.split("[")[1].split("]")[0]
                rel_pos = [float(coord.strip()) for coord in rel_pos_str.split(",")]
                key = tuple((node, ref_node))
                relative_positions[key] = rel_pos
            except (IndexError, ValueError) as e:
                pass

        positions = {}
        ref_node = next(iter(relative_positions))[1]
        positions[ref_node] = [0, 0]
        for node_c in relative_positions.keys():
            positions[node_c[0]] = relative_positions[node_c]

        return positions

    def predict_remaining_node_positions(self, common_nodes, positions, scene_graph):
        if len(common_nodes) < 2:
            raise ValueError("At least two common nodes are required to predict the position of other nodes")

        ref_points = sorted(list(common_nodes))[:2]
        ref_point_1, ref_point_2 = ref_points

        ref_pos_1 = np.array(scene_graph.nodes[ref_point_1]['position'])
        ref_pos_2 = np.array(scene_graph.nodes[ref_point_2]['position'])

        ref_vec_scene = ref_pos_2 - ref_pos_1
        try:
            ref_vec_subgraph = np.array(positions[ref_point_1]) - np.array(positions[ref_point_2])
        except KeyError as e:
            print(f"KeyError: {e}")
            return {}

        if np.allclose(ref_vec_subgraph, 0):
            ref_vec_subgraph = np.array([1000., 1000.])

        angle = np.arctan2(ref_vec_scene[1], ref_vec_scene[0]) - np.arctan2(ref_vec_subgraph[1], ref_vec_subgraph[0])
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

        scale_factor = np.linalg.norm(ref_vec_scene) / np.linalg.norm(ref_vec_subgraph)

        predicted_positions = {}
        for node, rel_pos in positions.items():
            if node not in ref_points:
                relative_position = np.array(rel_pos) - np.array(positions[ref_point_1])
                transformed_pos = np.dot(rotation_matrix, np.array(relative_position) * scale_factor) + ref_pos_1
                predicted_positions[node] = transformed_pos

        if len(predicted_positions) > 0:
            rel_pos_list = []
            for node, rel_pos in predicted_positions.items():
                rel_pos_list.append(rel_pos)
            position = sum(rel_pos_list) / len(rel_pos_list)
        else:
            rel_pos_list = [ref_pos_1, ref_pos_2]
            position = sum(rel_pos_list) / len(rel_pos_list)
        position = list(position)
        return position
