import os
import sys
import cv2
import numpy as np
import torch
import math
import dataclasses
import omegaconf
from openai import OpenAI
import base64
import re
from io import BytesIO
import supervision as sv
import random
from PIL import Image
from sklearn.cluster import DBSCAN  
from collections import Counter 
from omegaconf import DictConfig
from pathlib import PosixPath, Path
from supervision.draw.color import ColorPalette
import copy
import skimage

from .graphbuilder import GraphBuilder
from .goalgraphdecomposer import GoalGraphDecomposer
from .overlap import GraphMatcher
from .scenegraphcorrector import SceneGraphCorrector
from .utils.slam_classes import MapObjectList
from .utils.utils import filter_objects, gobs_to_detection_list
from .utils.mapping import compute_spatial_similarities, merge_detections_to_objects

# 超图推理模块
from ..reasoning.hypergraph_bp import HyperGraphReasoner

from ..utils.fmm.fmm_planner import FMMPlanner
from ..utils.fmm import pose_utils as pu
from ..utils.camera import get_camera_matrix
from ..utils.map import remove_small_frontiers
from ..utils.llm import LLM, VLM
from ..utils.depth import habitat_depth_to_metric
from ..utils.frontier_belief import (
    compute_arrival_reject_penalty,
    compute_frontier_belief_utility,
)
from ..utils.text_goal import format_text_goal_description

sys.path.append('third_party/Grounded-Segment-Anything/')
from grounded_sam_demo import load_model, get_grounding_output
import GroundingDINO.groundingdino.datasets.transforms as T
from segment_anything import sam_model_registry, SamPredictor
from lightglue import LightGlue, DISK
from lightglue.utils import match_pair , numpy_image_to_torch


ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
]


class RoomNode():
    """房间节点-表示一个房间空间"""
    def __init__(self, caption):
        self.caption = caption #房间名称
        self.exploration_level = 0 #探索程度
        self.nodes = set()  #包含物体节点集合
        self.group_nodes = [] #聚类后的物体组


class GroupNode():
    """物体组节点-表示一组空间上接近的物体节点"""
    def __init__(self, caption=''):
        self.caption = caption
        self.exploration_level = 0
        self.corr_score = 0
        self.center = None
        self.center_node = None
        self.nodes = []
        self.edges = set()
    
    def __lt__(self, other):
        return self.corr_score < other.corr_score
    
    def get_graph(self):
        """计算组的几何中心和中心节点，并生成组的文本描述"""
        self.center = np.array([node.center for node in self.nodes]).mean(axis=0)
        min_distance = np.inf
        for node in self.nodes:
            distance = np.linalg.norm(np.array(node.center) - np.array(self.center))
            if distance < min_distance:
                min_distance = distance
                self.center_node = node
            self.edges.update(node.edges)
        self.caption = self.graph_to_text(self.nodes, self.edges)

    def graph_to_text(self, nodes, edges):
        nodes_text = ', '.join([node.caption for node in nodes])
        edges_text = ', '.join([f"{edge.node1.caption} {edge.relation} {edge.node2.caption}" for edge in edges])
        return f"Nodes: {nodes_text}. Edges: {edges_text}."

class ObjectNode():
    """物体节点-表示一个具体的物体
    
    时序状态向量 τ_i = [t_first, t_last, n_obs, c_trend]
    - t_first: 节点首次被观测的时间戳
    - t_last: 节点最近一次被观测的时间戳  
    - n_obs: 节点的累计观测次数
    - c_trend: 节点置信度的移动平均趋势
    """
    def __init__(self):
        self.is_new_node = True
        self.caption = None
        self.object = None
        self.reason = None
        self.center = None
        self.room_node = None
        self.exploration_level = 0
        self.distance = 2
        self.score = 0.5
        self.edges = set()
        
        # ===== 时序状态向量 τ_i =====
        self.t_first = None           # 首次观测时间戳
        self.t_last = None            # 最近观测时间戳
        self.n_obs = 0                # 累计观测次数
        self.c_trend = 0.0            # 置信度移动平均趋势
        self.is_in_FOV = False        # 新增：节点是否在当前FOV内（局部条件状态机）
        self.confidence = 0.5         # 当前贝叶斯置信度
        self.confidence_history = []  # 置信度历史记录（用于计算趋势）
        self.observation_times = []   # 观测时间记录（用于贝叶斯计算）
        self.similarity_scores = []   # Backward-compatible evidence score used by confidence update.
        self.det_confidence_scores = []
        self.semantic_similarity_scores = []
        self.anomaly_checked = False  # ★ 新增：异常检查冷却标志

    def __lt__(self, other):
        return self.score < other.score

    def add_edge(self, edge):
        self.edges.add(edge)

    def remove_edge(self, edge):
        self.edges.discard(edge)
    
    def set_caption(self, new_caption):
        """设置新名称时，清除所有旧的边和状态，但保留时序信息"""
        for edge in list(self.edges):
            edge.delete()
        self.is_new_node = True
        self.caption = new_caption
        self.reason = None
        self.distance = 2
        self.score = 0.5
        self.exploration_level = 0
        self.edges.clear()
        # 注意：时序信息保留，因为物理实体未变
    
    def set_object(self, object):
        self.object = object
        self.object['node'] = self
    
    def set_center(self, center):
        self.center = center
    
    def update_temporal_state(self, current_time, similarity=None,
                              det_confidence=None, semantic_similarity=None):
        """更新节点的时序状态
        
        Args:
            current_time: 当前时间步
            similarity: 本次观测的语义相似度（可选）
        """
        if self.t_first is None:
            self.t_first = current_time
        self.t_last = current_time
        self.n_obs += 1
        self.anomaly_checked = False
        self.observation_times.append(current_time)
        
        evidence_score = similarity if similarity is not None else self.score
        self.similarity_scores.append(evidence_score)
        self.det_confidence_scores.append(
            det_confidence if det_confidence is not None else evidence_score
        )
        if semantic_similarity is not None:
            self.semantic_similarity_scores.append(semantic_similarity)

        MAX_OBS_HISTORY = 100
        if len(self.observation_times) > MAX_OBS_HISTORY:
            self.observation_times = self.observation_times[-MAX_OBS_HISTORY:]
            self.similarity_scores = self.similarity_scores[-MAX_OBS_HISTORY:]
            self.det_confidence_scores = self.det_confidence_scores[-MAX_OBS_HISTORY:]
            self.semantic_similarity_scores = self.semantic_similarity_scores[-MAX_OBS_HISTORY:]
    
    def get_temporal_vector(self):
        """获取时序状态向量 τ_i"""
        return [self.t_first, self.t_last, self.n_obs, self.c_trend]
    
    def get_time_span(self):
        """获取节点的观测时间跨度"""
        if self.t_first is None or self.t_last is None:
            return 0
        return self.t_last - self.t_first


class Edge():
    """边类 - 表示两个物体之间的空间关系
    
    时序边权重计算公式:
    w_ij(t) = (n_simultaneous / (n_obs_i + n_obs_j)) * exp(-λ * (t - t_last))
    - 共现频率: 两物体同时被观测的比例
    - 指数衰减: 遗忘过时的空间关系
    """
    # 边权重衰减系数
    LAMBDA_DECAY = 0.05
    
    def __init__(self, node1, node2, current_time=0):
        self.node1 = node1
        self.node2 = node2
        node1.add_edge(self)
        node2.add_edge(self)
        self.relation = None
        self.status = 'pending'
        self.confidence = 0.0
        self.evidence_count = 0
        self.reject_count = 0
        self.relation_votes = {}
        self.source = 'candidate'
        self.geometry_score = 0.0
        
        # ===== 时序边权重 =====
        self.t_create = current_time      # 边创建时间
        self.t_last = current_time        # 边最近更新时间
        self.n_simultaneous = 1           # 共现次数（初始为1）
        self.local_decay_ticks = 0        # 新增：局部条件激活的衰减时钟
        self.weight = 1.0                 # 时变权重

    def set_relation(self, relation):
        self.relation = relation

    def add_relation_evidence(self, relation, confidence, source='llm', current_time=None):
        self.relation = relation
        self.source = source
        self.evidence_count += 1
        self.relation_votes[relation] = self.relation_votes.get(relation, 0) + 1
        self.confidence = float(np.clip(max(self.confidence, confidence), 0.0, 1.0))
        if current_time is not None:
            self.t_last = current_time

    def reject(self):
        self.status = 'rejected'
        self.reject_count += 1
        self.confidence = 0.0

    def delete(self):
        self.node1.remove_edge(self)
        self.node2.remove_edge(self)

    def text(self):
        text = '({}, {}, {})'.format(self.node1.caption, self.node2.caption, self.relation)
        return text
    
    def update_temporal_weight(self, current_time):
        """更新边的时序权重
        局部条件激活衰减:Jaccard_Ratio * exp(-λ * local_decay_ticks
        
        Args:
            current_time: 当前时间步
        """
        # 如果边没有在当前帧共现，且至少一个端点本该在视野内（is_in_FOV），则衰减时钟走动
        if self.t_last < current_time:
            if self.node1.is_in_FOV or self.node2.is_in_FOV:
                self.local_decay_ticks += 1

        # 并集 = obs1 + obs2 - 交集(simultaneous)
        union_obs = self.node1.n_obs + self.node2.n_obs - self.n_simultaneous
        
        if union_obs > 0:
            cooccurrence_ratio = self.n_simultaneous / union_obs
        else:
            cooccurrence_ratio = 0.5
        
        # 局部条件指数衰减
        time_decay = np.exp(-self.LAMBDA_DECAY * self.local_decay_ticks)
        
        self.weight = cooccurrence_ratio * time_decay
        return self.weight
    
    def record_cooccurrence(self, current_time):
        """记录一次共现观测"""
        self.n_simultaneous += 1
        self.t_last = current_time
        self.evidence_count += 1
        if self.status != 'rejected':
            self.confidence = float(np.clip(self.confidence + 0.1, 0.0, 1.0))
    
    def check_temporal_paradox(self):
        """检测时间悖论：边的创建时间是否早于任一端点节点
        
        用于检测 LLM 推理导致的时序不一致
        """
        if self.node1.t_first is None or self.node2.t_first is None:
            return False
        node1_create = self.node1.t_first
        node2_create = self.node2.t_first
        return self.t_create < max(node1_create, node2_create)


class SubGraph():
    """子图类 - 表示以某个中心节点为核心的子图"""
    def __init__(self, center_node):
        self.center_node = center_node
        self.edges = self.center_node.edges
        self.center = self.center_node.center
        self.nodes = set()
        for edge in self.edges:
            self.nodes.add(edge.node1)
            self.nodes.add(edge.node2)

    def get_subgraph_2_text(self):
        text = ''
        edges = set()
        for node in self.nodes:
            text = text + node.caption + '/'
            edges.update(node.edges)
        text = text[:-1] + '\n'
        for edge in edges:
            text = text + edge.relation + '/'
        text = text[:-1]
        return text


class Graph():
    def __init__(self, args, is_navigation=True) -> None:
        self.args = args
        #地图参数
        self.map_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm
        self.map_size = args.map_size
        self.camera_matrix = get_camera_matrix(args.env_frame_width, args.env_frame_height, args.hfov)
        full_width, full_height = self.map_size, self.map_size
        self.full_width = full_width
        self.full_height = full_height
        self.visited = torch.zeros(full_width, full_height).float().cpu().numpy()
        self.device = args.device
        self.classes = ['item']
        self.BG_CLASSES = ["wall", "floor", "ceiling"]
        self.rooms = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.edge_list = []
        self.group_nodes = []
        self.init_room_nodes()
        self.is_navigation = is_navigation
        self.set_cfg()
        
        self.groundingdino_config_file = 'third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        self.groundingdino_checkpoint = 'data/models/groundingdino_swint_ogc.pth'
        self.bert_base_uncased_path = 'data/models/bert-base-uncased'  # BERT 模型本地路径，如果为空字符串则从 HuggingFace 下载
        self.sam_version = 'vit_h'
        self.sam_checkpoint = 'data/models/sam_vit_h_4b8939.pth'
        self.segment2d_results = []
        self.max_detections_per_object = 10
        self.threshold_list = {'bathtub': 3, 'bed': 3, 'cabinet': 2, 'chair': 1, 'chest_of_drawers': 3, 'clothes': 2, 'counter': 1, 'cushion': 3, 'fireplace': 3, 'gym_equipment': 2, 'picture': 3, 'plant': 3, 'seating': 0, 'shower': 2, 'sink': 2, 'sofa': 2, 'stool': 2, 'table': 1, 'toilet': 3, 'towel': 2, 'tv_monitor': 0}
        self.found_goal_times_threshold = 1
        self.N_max = 10
        self.node_space = 'table. tv. chair. cabinet. sofa. bed. windows. kitchen. bedroom. living room. mirror. plant. curtain. painting. picture'
        self.relations = ["next to", "opposite to", "below", "behind", "in front of", "on", "above"]
        self.relation_aliases = {
            "near": "next to",
            "nearby": "next to",
            "beside": "next to",
            "adjacent to": "next to",
            "close to": "next to",
            "under": "below",
            "underneath": "below",
            "in back of": "behind",
            "front of": "in front of",
            "facing": "opposite to",
            "opposite": "opposite to",
        }
        self.prompt_edge_proposal = '''
Provide the most possible single spatial relationship for each of the following object pairs. Answer with only one relationship per pair, and separate each answer with a newline character.
Examples:
Input:
Object pair(s):
(cabinet, chair)
Output:
next to
Input:
Object pair(s):
(table, lamp)
(bed, nightstand)
Output:
on
next to
Object pair(s):
        '''
        self.prompt_room_predict = 'Which room is the most likely to have the [{}] in: [{}]. Only answer the room.'
        self.prompt_graph_corr_0 = 'What is the probability of A and B appearing together. [A:{}], [B:{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'
        self.prompt_graph_corr_1 = 'What else do you need to know to determine the probability of A and B appearing together? [A:{}], [B:{}]. Please output a short question (output only one sentence with no additional text).'
        self.prompt_graph_corr_2 = 'Here is the objects and relationships near A: [{}] You answer the following question with a short sentence based on this information. Question: {}'
        self.prompt_graph_corr_3 = 'The probability of A and B appearing together is about {}. Based on the dialog: [{}], re-determine the probability of A and B appearing together. A:[{}], B:[{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'
        self.prompt_image2text = (
            'Identify the main object at the center of this image by its specific name '
            '(e.g., "toilet", "chair", "bed", NOT generic words like "object" or "item"). '
            'Then briefly describe the spatial relationships between this main object and '
            'other nearby objects. Start your answer with the object name directly.'
        )
        self.prompt_create_relation = """
Given the image, please analyze the spatial relationship between {obj1} and {obj2}.
If there is a clear spatial relationship, describe it using the following template:
"{obj1} and {obj2}: {obj1} is <relation type> {obj2}"
If no clear spatial relationship exists, state: "No clear spatial relationship between {obj1} and {obj2}"

Example output format for a relation:
"table and book: table is under book"

Example output format if no relation:
"No clear spatial relationship between table and book"

Please provide the relationship you can determine from the image.
        """
        self.grounded_sam = self.get_grounded_sam(self.device)
        self.llm = LLM(self.args.base_url, self.args.api_key, self.args.llm_model)
        self.vlm = VLM(self.args.base_url, self.args.api_key, self.args.vlm_model)
        self.graphbuilder = GraphBuilder(self.llm)
        self.goalgraphdecomposer = GoalGraphDecomposer(self.llm)
        self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
        self.image_matcher = LightGlue(features='disk').eval().to(self.device)
        
        # ===== 动态时序场景图参数初始化 =====
        self.navigate_steps = 0              # 当前导航步数（时间戳）
        self.last_visit_map = np.zeros((full_width, full_height))  # 位置访问时间记录
        self.temporal_graph_delta = None     # 图差分缓存
        self.T_forget = 100                  # 遗忘阈值（超过此时间未观测的节点标记为消失）
        
        # 时序参数配置
        self.temporal_config = {
            'alpha': 0.90,        # 贝叶斯置信度时间衰减系数（0.90加速早期收敛）
            'beta': 0.1,          # 时间跨度正则系数
            'gamma': 0.03,        # 最近未观测惩罚系数（适度加大遗忘压力）
            'conf_weight': 0.3,   # 前沿选择中置信度权重
            'novelty_weight': 0.2, # 前沿选择中新颖性权重
            'trend_threshold_up': 0.05,    # 置信度上升阈值
            'trend_threshold_down': -0.05, # 置信度下降阈值
            'active_confidence_threshold': 0.25,
            'anomaly_fresh_steps': 2,
        }
        
        # ===== 超图推理参数配置 =====
        self.hypergraph_config = {
            'bp_max_iterations': 15,       # BP 最大迭代次数
            'bp_damping': 0.5,             # 阻尼系数
            'clip_sim_threshold': 0.6,     # CLIP 相似度激活阈值
            'hyperedge_min_overlap': 0.34,  # 超边最小重叠比例
            'unary_weight': 1.0,           # 一元因子权重
            'binary_weight': 0.8,          # 二元因子权重
            'hyper_weight': 1.2,           # 高阶因子权重
            'unary_weight': 1.0,
            'prob_threshold': 0.12,
            'prob_keep_topk': 3,
            'use_hypergraph': True,        # 是否启用超图推理（False 则使用原 overlap）
        }
        
        # 初始化超图推理器（延迟初始化，等 LLM/VLM 就绪后）
        self.hypergraph_config['pending_edge_scale'] = 0.35
        self.edge_lifecycle_config = {
            'max_candidate_distance': 120.0,
            'max_candidates_per_new_node': 8,
            'min_geometry_score': 0.15,
            'initial_relation_confidence': 0.55,
            'min_active_confidence': 0.65,
            'min_active_evidence': 2,
            'pending_export_scale': 0.35,
        }
        self.hypergraph_reasoner = None
        self.prob_map = {}  # BP 推断输出的后验概率分布
        self.node_uid_map = {}
        self.scenegraph_uid_to_node = {}
        self.scenegraph_edge_key_to_edge = {}
        
        # ===== 目标图属性初始化 =====
        self.goalgraph = None
        self.goalgraph_decomposed = {}
        self.text_goal = None
        self.current_goat_goal_type = None
        
        # ===== 门口窥视（Doorway Peek）参数配置 =====
        self.peek_config = {
            'enabled': True,                # 是否启用门口窥视
            'min_move_dist': 30,            # 两次窥视之间最小移动距离（地图格数）
            'irrelevant_penalty': 5.0,      # 不相关区域前沿惩罚权重
            'penalty_radius': 80,           # 惩罚影响半径（地图格数）
            'relevance_cache_size': 50,     # 房间-目标相关性缓存大小
        }
        self.prompt_room_classify = (
            'Choose exactly one room/area label from this list: '
            'bedroom, bathroom, kitchen, living room, dining room, office, hallway, '
            'closet, laundry room, garage, unknown. '
            'Return only the label, no sentence. If uncertain, return unknown.'
        )
        self.prompt_room_relevance = (
            'Is a "{room_type}" likely to contain a "{target}"? '
            'Answer only "yes" or "no".'
        )
        # 窥视状态（在 reset() 中也会重置）
        self._peek_last_pos = None            # 上次窥视时的机器人位置
        self._peeked_irrelevant_regions = []  # [(x, y, room_type), ...] 不相关区域记录
        self._room_relevance_cache = {}       # {(room_type, target): bool} 缓存

        self._peeked_irrelevant_regions = []
        self._room_relevance_cache = {}

        # GOAT is long-horizon: preserve observed object locations across
        # subtasks and normalize common category aliases for recall.
        self.goal_memory = {}
        self.goal_aliases = {
            'fridge': 'refrigerator',
            'freezer': 'refrigerator',
            'refrigerator': 'refrigerator',
            'couch': 'sofa',
            'settee': 'sofa',
            'sofa': 'sofa',
            'television': 'tv',
            'tv monitor': 'tv',
            'tvmonitor': 'tv',
            'monitor': 'tv',
            'tv': 'tv',
            'potted plant': 'plant',
            'decorative plant': 'plant',
            'flowerpot': 'plant',
            'plant': 'plant',
            'flower vase': 'vase',
            'vase': 'vase',
            'dining table': 'table',
            'coffee table': 'table',
            'side table': 'table',
            'table': 'table',
            'night stand': 'nightstand',
            'bedside table': 'nightstand',
            'bedside': 'nightstand',
            'nightstand': 'nightstand',
            'book case': 'bookshelf',
            'bookcase': 'bookshelf',
            'shelving': 'shelf',
            'bookshelf': 'bookshelf',
            'cupboard': 'cabinet',
            'cabinetry': 'cabinet',
            'drawer': 'cabinet',
            'cabinet': 'cabinet',
            'dresser': 'dresser',
            'chest of drawers': 'dresser',
            'microwave oven': 'microwave',
            'stove': 'oven',
        }

        self.frontier_config = {
            'min_distance_m': 0.8,
            'max_distance_m': 30.0,
            'distance_weight': 4.0,
            'semantic_goal_weight': 4.0,
            'cooldown_penalty': 4.0,
            'cooldown_radius': 14.0,
            'cooldown_steps': 80,
            'stall_replans': 4,
            'progress_eps_m': 0.35,
            'direct_goal_radius_m': 8.0,
            'goat_direct_goal_radius_m': 40.0,
            'memory_direct_goal_radius_m': 40.0,
            'direct_goal_snap_max_m': 1.0,
            'semantic_goal_min_conf': 0.15,
            'rescue_patience': 3,
            'rescue_min_distance_m': 0.3,
            'peek_penalty_limit': 2.5,
            'exhausted_frontier_penalty': float(
                getattr(args, 'exhausted_frontier_penalty', 6.0)
            ),
            'exhausted_frontier_radius_m': float(
                getattr(args, 'exhausted_frontier_radius_m', 2.0)
            ),
            'exhausted_frontier_ttl_steps': int(
                getattr(args, 'exhausted_frontier_ttl_steps', 320)
            ),
            'exhausted_frontier_max_regions': int(
                getattr(args, 'exhausted_frontier_max_regions', 80)
            ),
            'arrival_reject_enabled': bool(
                getattr(args, 'arrival_reject_frontier_enabled', True)
            ),
            'arrival_reject_penalty': float(
                getattr(args, 'arrival_reject_frontier_penalty', 3.0)
            ),
            'arrival_reject_radius_m': float(
                getattr(args, 'arrival_reject_frontier_radius_m', 2.0)
            ),
            'belief_enabled': not bool(getattr(args, 'disable_frontier_belief', False)),
            'belief_prior': float(getattr(args, 'frontier_belief_prior', 0.35)),
            'belief_weight': float(getattr(args, 'frontier_belief_weight', 1.5)),
            'belief_positive_logit_weight': float(
                getattr(args, 'frontier_belief_positive_weight', 1.6)
            ),
            'belief_negative_logit_weight': float(
                getattr(args, 'frontier_belief_negative_weight', 1.8)
            ),
            'belief_distance_cost_weight': float(
                getattr(args, 'frontier_belief_distance_cost', 0.25)
            ),
            'debug_topk': 5,
        }
        self.rejected_goal_regions = []
        self._reset_frontier_state()
        self._reset_semantic_goal_state()

        self.room_type_candidates = [
            'bedroom', 'bathroom', 'kitchen', 'living room', 'dining room',
            'office', 'hallway', 'closet', 'laundry room', 'garage', 'unknown'
        ]
        self.room_object_priors = {
            'chair': {'living room', 'dining room', 'office', 'bedroom', 'kitchen'},
            'sofa': {'living room', 'bedroom'},
            'bed': {'bedroom'},
            'toilet': {'bathroom'},
            'sink': {'bathroom', 'kitchen', 'laundry room'},
            'tv': {'living room', 'bedroom'},
            'cabinet': {'kitchen', 'bathroom', 'bedroom', 'living room'},
            'table': {'dining room', 'living room', 'office', 'kitchen'},
            'plant': {'living room', 'office', 'bedroom', 'dining room'},
        }
        self.passthrough_room_types = {'hallway', 'corridor', 'passage', 'entryway', 'foyer', 'lobby'}

        # ===== 消融实验开关：将 args.disable_* 标志应用到各模块配置 =====
        # 默认值为 False（不禁用），保证未传入 flag 时所有创新机制均启用
        self.disable_temporal   = getattr(args, 'disable_temporal',   False)
        self.disable_hypergraph = getattr(args, 'disable_hypergraph', False)
        self.disable_peek       = getattr(args, 'disable_peek',       False)
        self.disable_arrival_reject = getattr(args, 'disable_arrival_reject', False)
        self.disable_frontier_belief = getattr(args, 'disable_frontier_belief', False)

        if self.disable_temporal:
            # 关闭时序场景图：置信度固定为 0.5，时序评分权重归零
            self.temporal_config['conf_weight']    = 0.0
            self.temporal_config['novelty_weight'] = 0.0
            print('[Ablation] Temporal scene graph DISABLED.')

        if self.disable_hypergraph:
            # 关闭超图推理：回退到原始 overlap() 硬匹配
            self.hypergraph_config['use_hypergraph'] = False
            print('[Ablation] Hypergraph reasoning DISABLED (using overlap fallback).')

        if self.disable_peek:
            # 关闭门口窥视：不对前沿施加语义房间惩罚
            self.peek_config['enabled'] = False
            print('[Ablation] Doorway peek DISABLED.')

        if self.disable_arrival_reject:
            self.frontier_config['arrival_reject_enabled'] = False
            print('[Ablation] Arrival-reject frontier feedback DISABLED.')

        if self.disable_frontier_belief:
            self.frontier_config['belief_enabled'] = False
            print('[Ablation] Frontier belief utility DISABLED.')

    def set_cfg(self):
        cfg = {'dataset_config': PosixPath('tools/replica.yaml'), 'scene_id': 'room0', 'start': 0, 'end': -1, 'stride': 5, 'image_height': 680, 'image_width': 1200, 'gsa_variant': 'none', 'detection_folder_name': 'gsa_detections_${gsa_variant}', 'det_vis_folder_name': 'gsa_vis_${gsa_variant}', 'color_file_name': 'gsa_classes_${gsa_variant}', 'device': 'cuda', 'use_iou': True, 'spatial_sim_type': 'overlap', 'phys_bias': 0.0, 'match_method': 'sim_sum', 'semantic_threshold': 0.5, 'physical_threshold': 0.5, 'sim_threshold': 1.2, 'use_contain_number': False, 'contain_area_thresh': 0.95, 'contain_mismatch_penalty': 0.5, 'mask_area_threshold': 25, 'mask_conf_threshold': 0.95,
               'max_bbox_area_ratio': 0.5, 'skip_bg': True, 'min_points_threshold': 16, 'downsample_voxel_size': 0.025, 'dbscan_remove_noise': True, 'dbscan_eps': 0.1, 'dbscan_min_points': 10, 'obj_min_points': 0, 'obj_min_detections': 3, 'merge_overlap_thresh': 0.7, 'merge_visual_sim_thresh': 0.8, 'merge_text_sim_thresh': 0.8, 'denoise_interval': 20, 'filter_interval': -1, 'merge_interval': 20, 'save_pcd': True, 'save_suffix': 'overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub', 'vis_render': False, 'debug_render': False, 'class_agnostic': True, 'save_objects_all_frames': True, 'render_camera_path': 'replica_room0.json', 'max_num_points': 512}
        cfg = DictConfig(cfg)
        if self.is_navigation:
            cfg.sim_threshold = 0.8
            cfg.sim_threshold_spatial = 0.01
        self.cfg = cfg

    def set_agent(self, agent):
        self.agent = agent

    def set_navigate_steps(self, navigate_steps):
        self.navigate_steps = navigate_steps

    def set_room_map(self, room_map):
        self.room_map = room_map

    def set_fbe_free_map(self, fbe_free_map):
        self.fbe_free_map = fbe_free_map
    
    def set_observations(self, observations):
        self.observations = observations
        self.image_rgb = observations['rgb'].copy()
        self.image_depth = observations['depth'].copy()
        self.pose_matrix = self.get_pose_matrix()

    def set_obj_goal(self, obj_goal):
        self.obj_goal = obj_goal

    def set_open_vocab_categories(self, categories):
        """Extend GroundingDINO's vocabulary without adding BEV channels."""
        labels = [item.strip() for item in self.node_space.split('.') if item.strip()]
        seen = {self._canonical_label(item) for item in labels}
        for category in categories or []:
            label = self._canonical_label(category)
            if label and label not in seen:
                labels.append(label)
                seen.add(label)
        self.node_space = '. '.join(labels) + '.'
        print(f'[OpenVocab] GroundingDINO vocabulary: {len(labels)} labels')
    #从图像目标生成目标图
    def set_image_goal(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        self.instance_imagegoal = image
        text_goal = self.vlm(self.prompt_image2text, self.instance_imagegoal)
        self.set_text_goal(text_goal)
    #从文本目标构建目标图
    def set_text_goal(self, text_goal):
        text_goal = format_text_goal_description(text_goal)
        self.text_goal = text_goal
        self.goalgraph = self.graphbuilder.build_graph_from_text(text_goal)
        self.goalgraph_decomposed = self.goalgraphdecomposer.goal_decomposition(self.goalgraph)

    def set_frontier_map(self, frontier_map):
        self.frontier_map = frontier_map

    def set_full_map(self, full_map):
        self.full_map = full_map
        h, w = int(full_map.shape[-2]), int(full_map.shape[-1])
        self.full_height, self.full_width = h, w
        if getattr(self, 'last_visit_map', None) is None or self.last_visit_map.shape != (h, w):
            self.last_visit_map = np.zeros((h, w))

    def set_full_pose(self, full_pose):
        self.full_pose = full_pose

    def _full_map_shape(self):
        if hasattr(self, 'full_map') and self.full_map is not None:
            return int(self.full_map.shape[-2]), int(self.full_map.shape[-1])
        return int(self.full_height), int(self.full_width)

    def _to_rc(self, point):
        if point is None:
            return None
        arr = np.asarray(point, dtype=float).reshape(-1)
        if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
            return None
        return [int(round(float(arr[0]))), int(round(float(arr[1])))]

    def _in_bounds_rc(self, point, shape=None):
        rc = self._to_rc(point)
        if rc is None:
            return False
        h, w = shape if shape is not None else self._full_map_shape()
        return 0 <= rc[0] < h and 0 <= rc[1] < w

    def _classify_global_goal(self, point):
        rc = self._to_rc(point)
        if rc is None:
            return 'invalid'
        h, w = self._full_map_shape()
        if not (0 <= rc[0] < h and 0 <= rc[1] < w):
            return 'out_of_global_map'
        if not hasattr(self, 'full_map') or self.full_map is None:
            return 'in_map_unknown'
        obstacle = self.full_map[0, 0, rc[0], rc[1]] > 0
        explored = self.full_map.shape[1] > 1 and self.full_map[0, 1, rc[0], rc[1]] > 0
        if bool(obstacle.item() if hasattr(obstacle, 'item') else obstacle):
            return 'in_map_obstacle'
        if bool(explored.item() if hasattr(explored, 'item') else explored):
            return 'in_map_explored_free'
        return 'in_map_unknown'

    def _global_rc_to_flipped_fmm_state(self, point, traversible_shape):
        rc = self._to_rc(point)
        if rc is None:
            return None
        row, col = rc
        state = [int(traversible_shape[0] - row - 2), int(col + 1)]
        if 0 <= state[0] < traversible_shape[0] and 0 <= state[1] < traversible_shape[1]:
            return state
        return None

    def _flipped_fmm_state_to_global_rc(self, state, traversible_shape):
        if state is None:
            return None
        return [
            int(traversible_shape[0] - int(state[0]) - 2),
            int(int(state[1]) - 1),
        ]

    def _reset_frontier_state(self):
        self._frontier_cooldowns = []
        self._frontier_exhausted_regions = []
        self._last_frontier_goal = None
        self._last_frontier_dist_m = None
        self._last_frontier_step = None
        self._frontier_stall_count = 0
        self._empty_frontier_streak = 0
        self._frontier_rescue_count = 0
        self.frontier_locations = np.empty((0, 2), dtype=np.int32)
        self.frontier_locations_16 = np.empty((0, 2), dtype=np.int32)
        self.scores = np.zeros(0, dtype=np.float32)
        self.frontier_debug = {}
        self.frontier_candidates_debug = []

    def _reset_semantic_goal_state(self):
        self._semantic_goal_source = 'none'
        self._semantic_goal_confidence = 0.0
        self._semantic_direct_allowed = False
        self._semantic_goal_selected_direct = False

    def _set_semantic_goal_state(self, source, confidence=0.0, direct_allowed=False):
        self._semantic_goal_source = source
        self._semantic_goal_confidence = float(confidence or 0.0)
        self._semantic_direct_allowed = bool(direct_allowed)

    def set_rejected_goal_regions(self, rejected_goals):
        """Sync arrival-verifier negative evidence into frontier selection."""
        regions = []
        for goal in rejected_goals or []:
            rc = self._to_rc(goal)
            if self._in_bounds_rc(rc):
                regions.append([int(rc[0]), int(rc[1])])
        self.rejected_goal_regions = regions

    def _canonical_label(self, text):
        text = str(text or '').lower().replace('_', ' ').replace('-', ' ')
        text = re.sub(r'[^a-z0-9 ]+', ' ', text)
        return ' '.join(text.split())

    def _goal_memory_key(self, text):
        """Return the canonical key used for GOAT target recall."""
        label = self._canonical_label(text)
        if not label:
            return ''
        aliases = getattr(self, 'goal_aliases', {})
        if label in aliases:
            return aliases[label]
        tokens = label.split()
        if len(tokens) == 1:
            token = tokens[0]
            if token in aliases:
                return aliases[token]
            if token.endswith('ies') and len(token) > 4:
                return token[:-3] + 'y'
            if token.endswith('s') and len(token) > 3:
                singular = token[:-1]
                return aliases.get(singular, singular)
            return token
        # Prefer a known alias contained in a longer caption, e.g. "white fridge".
        for width in range(min(4, len(tokens)), 0, -1):
            for start in range(0, len(tokens) - width + 1):
                phrase = ' '.join(tokens[start:start + width])
                if phrase in aliases:
                    return aliases[phrase]
        return label

    def _goal_labels_match(self, left, right):
        left_key = self._goal_memory_key(left)
        right_key = self._goal_memory_key(right)
        return bool(left_key and right_key and left_key == right_key)

    def _remember_goal_node(self, node):
        if getattr(self.args, 'goal_type', '') != 'goat':
            return
        if node is None or node.center is None or not self._in_bounds_rc(node.center):
            return
        key = self._goal_memory_key(getattr(node, 'caption', ''))
        if not key:
            return
        memory = getattr(self, 'goal_memory', {})
        entries = memory.setdefault(key, [])
        entry = {
            'node': node,
            'caption': node.caption,
            'center': [int(node.center[0]), int(node.center[1])],
            'confidence': float(getattr(node, 'confidence', 0.5)),
            'last_seen': int(getattr(node, 't_last', self.navigate_steps) or self.navigate_steps),
            'n_obs': int(getattr(node, 'n_obs', 0) or 0),
        }
        for idx, old in enumerate(entries):
            if old.get('node') is node:
                entries[idx] = entry
                break
        else:
            entries.append(entry)
        entries.sort(
            key=lambda item: (
                float(item.get('confidence', 0.0)),
                int(item.get('n_obs', 0)),
                int(item.get('last_seen', -1)),
            ),
            reverse=True,
        )
        del entries[8:]
        self.goal_memory = memory

    def _update_goal_memory_from_nodes(self):
        if getattr(self.args, 'goal_type', '') != 'goat':
            return
        for node in getattr(self, 'nodes', []):
            self._remember_goal_node(node)

    def recall_current_goal_from_memory(self, min_confidence=0.05):
        """Recall the best observed location for the current GOAT goal."""
        if getattr(self.args, 'goal_type', '') != 'goat':
            return None
        if getattr(self, 'current_goat_goal_type', None) == 'image':
            # GOAT image goals are instance-specific. Category-level scene
            # memory can be used as context later, but should not directly
            # recall a target location without image-instance verification.
            return None
        target = getattr(self, 'obj_goal', None)
        key = self._goal_memory_key(target)
        if not key:
            return None
        entries = list(getattr(self, 'goal_memory', {}).get(key, []))
        valid = []
        for entry in entries:
            center = entry.get('center')
            if center is None or not self._in_bounds_rc(center):
                continue
            confidence = float(entry.get('confidence', 0.0))
            if confidence < float(min_confidence):
                continue
            valid.append(entry)
        if not valid:
            return None
        valid.sort(
            key=lambda item: (
                float(item.get('confidence', 0.0)),
                int(item.get('n_obs', 0)),
                int(item.get('last_seen', -1)),
            ),
            reverse=True,
        )
        best = valid[0]
        self._set_semantic_goal_state(
            'goat_memory',
            max(float(best.get('confidence', 0.0)), 0.55),
            True,
        )
        print(
            f"[GOATMemory] recall target={target} key={key} "
            f"caption={best.get('caption')} center={best.get('center')} "
            f"conf={float(best.get('confidence', 0.0)):.2f} "
            f"last_seen={best.get('last_seen')}"
        )
        return list(best['center'])

    def _normalize_room_type(self, raw_room):
        text = self._canonical_label(raw_room)
        if not text:
            return 'unknown'

        candidates = getattr(self, 'room_type_candidates', [])
        for room in candidates:
            if text == room:
                return room

        keyword_rules = [
            ('bathroom', ['bathroom', 'toilet', 'shower', 'bathtub', 'bath tub', 'sink']),
            ('kitchen', ['kitchen', 'refrigerator', 'fridge', 'freezer', 'stove', 'oven', 'microwave']),
            ('bedroom', ['bedroom', 'bed ', ' bed', 'mattress', 'nightstand', 'dresser']),
            ('living room', ['living room', 'living space', 'sofa', 'couch', 'television', ' tv ']),
            ('dining room', ['dining room', 'dining table']),
            ('office', ['office', 'desk', 'computer', 'workstation']),
            ('hallway', ['hallway', 'corridor', 'doorway', 'passage', 'entryway', 'foyer']),
            ('laundry room', ['laundry', 'washer', 'dryer']),
            ('closet', ['closet', 'wardrobe']),
            ('garage', ['garage']),
        ]
        padded = f' {text} '
        for room, keywords in keyword_rules:
            if any(keyword in padded for keyword in keywords):
                return room
        return 'unknown'

    def _target_room_prior_relevance(self, room_type, target):
        room_type = self._normalize_room_type(room_type)
        target = self._canonical_label(target)
        if room_type == 'unknown' or not target:
            return None
        if room_type in getattr(self, 'passthrough_room_types', set()):
            return True
        for key, rooms in getattr(self, 'room_object_priors', {}).items():
            if key in target.split() or target == key:
                return room_type in rooms
        return None

    def _node_detection_confidence(self, object_data, fallback=0.5):
        conf = object_data.get('conf', fallback) if object_data is not None else fallback
        if isinstance(conf, (list, tuple, np.ndarray)):
            if len(conf) == 0:
                return float(fallback)
            conf = np.mean(conf)
        try:
            return float(np.clip(conf, 0.0, 1.0))
        except Exception:
            return float(fallback)

    def _node_is_active(self, node, current_time=None, threshold=None, require_center=False):
        if node is None:
            return False
        if require_center and (node.center is None or not self._in_bounds_rc(node.center)):
            return False
        if current_time is None:
            current_time = self.navigate_steps
        if threshold is None:
            threshold = self.temporal_config.get('active_confidence_threshold', 0.25)
        if node.t_last is None:
            return False
        if current_time - node.t_last > self.T_forget:
            return False
        if float(getattr(node, 'confidence', 0.0)) < float(threshold):
            return False
        return True

    def _active_nodes(self, current_time=None, threshold=None, require_center=False):
        return [
            node for node in self.nodes
            if self._node_is_active(node, current_time, threshold, require_center)
        ]

    def _refresh_node_confidences(self, current_time):
        max_conf_history = 50
        for node in self.nodes:
            if self.disable_temporal:
                node.confidence = 0.5
            else:
                node.confidence = self.compute_bayesian_confidence(node, current_time)
            node.confidence_history.append(node.confidence)
            if len(node.confidence_history) > max_conf_history:
                node.confidence_history = node.confidence_history[-max_conf_history:]

            if len(node.confidence_history) > 1:
                window = min(10, len(node.confidence_history))
                recent_conf = node.confidence_history[-window:]
                node.c_trend = np.mean(np.diff(recent_conf)) if len(recent_conf) > 1 else 0.0

    def _update_current_visit(self):
        if not hasattr(self, 'observations') or self.observations is None:
            return
        if 'gps' not in self.observations:
            return
        map_h, map_w = self._full_map_shape()
        current_pose_col = map_w / 2.0 + (self.observations['gps'][0] * 100.0 / self.map_resolution)
        current_pose_row_world = map_h / 2.0 - (self.observations['gps'][1] * 100.0 / self.map_resolution)
        current_pose_row = map_h - 1 - current_pose_row_world
        self.update_visit_map((current_pose_row, current_pose_col), self.navigate_steps)

    def _frontier_node_evidence(self, frontier, radius=80, max_nodes=3):
        entries = []
        for node in self._active_nodes(require_center=True):
            dist = float(np.linalg.norm(np.asarray(frontier) - np.asarray(node.center)))
            if dist >= radius:
                continue
            uid = getattr(self, 'node_uid_map', {}).get(node)
            entries.append({
                'caption': node.caption,
                'dist': round(dist, 1),
                'conf': round(float(getattr(node, 'confidence', 0.0)), 3),
                'prob': round(float(getattr(self, 'prob_map', {}).get(uid, 0.0)), 3) if uid else 0.0,
            })
        entries.sort(key=lambda item: (-item['prob'], -item['conf'], item['dist']))
        return entries[:max_nodes]

    def _empty_frontier_result(self, reason, num_raw=0, **extra):
        self._empty_frontier_streak = getattr(self, '_empty_frontier_streak', 0) + 1
        self.frontier_locations = np.empty((0, 2), dtype=np.int32)
        self.frontier_locations_16 = np.empty((0, 2), dtype=np.int32)
        self.scores = np.zeros(0, dtype=np.float32)
        self.frontier_debug = {
            'reason': reason,
            'num_raw': int(num_raw),
            'empty_frontier_streak': int(self._empty_frontier_streak),
        }
        self.frontier_candidates_debug = []
        for key, value in extra.items():
            if isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            self.frontier_debug[key] = value
        print(f'[Frontier] {reason}: {self.frontier_debug}')
        return None

    def _expire_frontier_cooldowns(self):
        now = getattr(self, 'navigate_steps', 0)
        self._frontier_cooldowns = [
            item for item in getattr(self, '_frontier_cooldowns', [])
            if item.get('until', 0) > now
        ]

    def _expire_frontier_exhausted_regions(self):
        now = getattr(self, 'navigate_steps', 0)
        self._frontier_exhausted_regions = [
            item for item in getattr(self, '_frontier_exhausted_regions', [])
            if item.get('until', 0) > now
        ]

    def _mark_frontier_exhausted(self, goal_rc, reason):
        rc = self._to_rc(goal_rc)
        if rc is None:
            return
        cfg = self.frontier_config
        ttl_steps = int(cfg.get('exhausted_frontier_ttl_steps', 0))
        penalty = float(cfg.get('exhausted_frontier_penalty', 0.0))
        if ttl_steps <= 0 or penalty <= 0:
            return
        self._expire_frontier_exhausted_regions()
        now = getattr(self, 'navigate_steps', 0)
        radius_cells = (
            float(cfg.get('exhausted_frontier_radius_m', 2.0))
            * 100.0 / float(self.map_resolution)
        )
        for item in self._frontier_exhausted_regions:
            center = np.asarray(item['center'], dtype=float)
            if np.linalg.norm(np.asarray(rc, dtype=float) - center) <= radius_cells:
                item['center'] = rc
                item['until'] = now + ttl_steps
                item['reason'] = reason
                item['count'] = int(item.get('count', 1)) + 1
                return
        self._frontier_exhausted_regions.append({
            'center': rc,
            'until': now + ttl_steps,
            'reason': reason,
            'count': 1,
        })
        max_regions = int(cfg.get('exhausted_frontier_max_regions', 80))
        if max_regions > 0 and len(self._frontier_exhausted_regions) > max_regions:
            self._frontier_exhausted_regions = self._frontier_exhausted_regions[-max_regions:]

    def _cooldown_frontier(self, goal_rc, reason):
        rc = self._to_rc(goal_rc)
        if rc is None:
            return
        cfg = self.frontier_config
        self._frontier_cooldowns.append({
            'center': rc,
            'until': self.navigate_steps + int(cfg['cooldown_steps']),
            'reason': reason,
        })
        if reason in ('no_progress', 'reached', 'unreachable'):
            self._mark_frontier_exhausted(rc, reason)
        print(f"[Frontier] cooldown {rc} ({reason}) until step {self._frontier_cooldowns[-1]['until']}")

    def _frontier_cooldown_mask(self, frontier_goals):
        self._expire_frontier_cooldowns()
        if not self._frontier_cooldowns or len(frontier_goals) == 0:
            return np.zeros(len(frontier_goals), dtype=bool)
        radius = float(self.frontier_config['cooldown_radius'])
        goals = np.asarray(frontier_goals, dtype=float)
        mask = np.zeros(len(goals), dtype=bool)
        for item in self._frontier_cooldowns:
            center = np.asarray(item['center'], dtype=float)
            mask |= np.linalg.norm(goals - center, axis=1) <= radius
        return mask

    def _update_frontier_progress(self, fmm_dist_flipped):
        last_goal = self._to_rc(getattr(self, '_last_frontier_goal', None))
        if last_goal is None:
            return
        now = getattr(self, 'navigate_steps', 0)
        last_step = getattr(self, '_last_frontier_step', None)
        if last_step is not None and now <= last_step:
            return
        idx = [last_goal[0] + 1, last_goal[1] + 1]
        if not (0 <= idx[0] < fmm_dist_flipped.shape[0] and 0 <= idx[1] < fmm_dist_flipped.shape[1]):
            self._cooldown_frontier(last_goal, 'out_of_fmm_bounds')
            self._last_frontier_goal = None
            self._last_frontier_dist_m = None
            self._frontier_stall_count = 0
            return

        dist_m = float(fmm_dist_flipped[idx[0], idx[1]] / 20.0)
        if not np.isfinite(dist_m):
            self._cooldown_frontier(last_goal, 'unreachable')
            self._last_frontier_goal = None
            self._last_frontier_dist_m = None
            self._frontier_stall_count = 0
            return

        if dist_m <= float(self.frontier_config['min_distance_m']):
            self._cooldown_frontier(last_goal, 'reached')
            self._last_frontier_goal = None
            self._last_frontier_dist_m = None
            self._frontier_stall_count = 0
            return

        last_dist = self._last_frontier_dist_m
        progress_eps = float(self.frontier_config['progress_eps_m'])
        if last_dist is not None and dist_m >= float(last_dist) - progress_eps:
            self._frontier_stall_count += 1
        else:
            self._frontier_stall_count = 0

        if self._frontier_stall_count >= int(self.frontier_config['stall_replans']):
            self._cooldown_frontier(last_goal, 'no_progress')
            self._last_frontier_goal = None
            self._last_frontier_dist_m = None
            self._frontier_stall_count = 0
        else:
            self._last_frontier_dist_m = dist_m

    def _snap_goal_to_reachable(self, goal_rc, traversible, current_fmm_dist):
        rc = self._to_rc(goal_rc)
        if rc is None:
            return None, None
        state = self._global_rc_to_flipped_fmm_state(rc, traversible.shape)
        if state is None:
            return None, None

        max_dist = float(np.nanmax(current_fmm_dist))

        def _is_reachable(r, c):
            return (
                0 <= r < traversible.shape[0]
                and 0 <= c < traversible.shape[1]
                and traversible[r, c] > 0
                and current_fmm_dist[r, c] < max_dist - 1e-5
            )

        if _is_reachable(state[0], state[1]):
            return rc, float(current_fmm_dist[state[0], state[1]] / 20.0)

        max_radius = 35
        sr, sc = state
        best = None
        best_cost = float('inf')
        for radius in range(1, max_radius + 1):
            r1, r2 = max(0, sr - radius), min(traversible.shape[0], sr + radius + 1)
            c1, c2 = max(0, sc - radius), min(traversible.shape[1], sc + radius + 1)
            rows, cols = np.where(
                (traversible[r1:r2, c1:c2] > 0)
                & (current_fmm_dist[r1:r2, c1:c2] < max_dist - 1e-5)
            )
            if len(rows) == 0:
                continue
            rows = rows + r1
            cols = cols + c1
            costs = np.hypot(rows - sr, cols - sc) + 0.05 * current_fmm_dist[rows, cols]
            idx = int(np.argmin(costs))
            if float(costs[idx]) < best_cost:
                best_cost = float(costs[idx])
                best = [int(rows[idx]), int(cols[idx])]
                break
        if best is None:
            return None, None
        snapped = self._flipped_fmm_state_to_global_rc(best, traversible.shape)
        return snapped, float(current_fmm_dist[best[0], best[1]] / 20.0)

    def _normalize_inverse_distance(self, distances_m, min_m=None, max_m=None):
        if len(distances_m) == 0:
            return np.zeros(0)
        if min_m is None:
            min_m = float(np.nanmin(distances_m))
        if max_m is None:
            max_m = float(np.nanmax(distances_m))
        denom = max(float(max_m) - float(min_m), 1e-6)
        return 1.0 - np.clip((distances_m - float(min_m)) / denom, 0.0, 1.0)

    def _ensure_edge_metadata(self, edge):
        defaults = {
            'status': 'active' if edge.relation is not None else 'pending',
            'confidence': max(float(getattr(edge, 'weight', 0.5)), 0.5) if edge.relation is not None else 0.0,
            'evidence_count': 1 if edge.relation is not None else 0,
            'reject_count': 0,
            'relation_votes': {},
            'source': 'legacy' if edge.relation is not None else 'candidate',
            'geometry_score': 0.0,
        }
        for key, value in defaults.items():
            if not hasattr(edge, key):
                setattr(edge, key, value)

    def _edge_key(self, node1, node2):
        return tuple(sorted([id(node1), id(node2)]))

    def _find_edge(self, node1, node2):
        for edge in node1.edges:
            if (edge.node1 is node1 and edge.node2 is node2) or (edge.node1 is node2 and edge.node2 is node1):
                return edge
        return None

    def _normalize_relation(self, relation):
        if relation is None:
            return None
        text = str(relation).lower().strip().strip('"').strip("'").strip()
        text = text.replace('_', ' ').replace('.', '').replace(',', '')
        if not text:
            return None
        no_relation_tokens = ['no clear', 'no relation', 'none', 'unknown', 'cannot determine', 'not enough']
        if any(token in text for token in no_relation_tokens):
            return None
        if ':' in text:
            text = text.split(':')[-1].strip()
        if ' is ' in text:
            text = text.split(' is ', 1)[1].strip()
        for alias, canonical in sorted(self.relation_aliases.items(), key=lambda item: len(item[0]), reverse=True):
            if alias == text or alias in text:
                return canonical
        for relation_name in sorted(self.relations, key=len, reverse=True):
            if relation_name == text or relation_name in text:
                return relation_name
        return None

    def _edge_geometry_score(self, node1, node2, current_observed_nodes=None):
        if node1.center is None or node2.center is None:
            return 0.0
        if not self._in_bounds_rc(node1.center) or not self._in_bounds_rc(node2.center):
            return 0.0
        max_dist = float(self.edge_lifecycle_config['max_candidate_distance'])
        dist = float(np.linalg.norm(np.array(node1.center) - np.array(node2.center)))
        score = max(0.0, 1.0 - dist / max_dist)
        if current_observed_nodes is not None and node1 in current_observed_nodes and node2 in current_observed_nodes:
            score += 0.25
        if node1.room_node is not None and node1.room_node is node2.room_node:
            score += 0.15
        return float(np.clip(score, 0.0, 1.0))

    def _select_edge_candidates(self, new_node, old_nodes, current_observed_nodes):
        scored = []
        min_score = float(self.edge_lifecycle_config['min_geometry_score'])
        for old_node in old_nodes:
            if old_node is new_node or self._find_edge(new_node, old_node) is not None:
                continue
            score = self._edge_geometry_score(new_node, old_node, current_observed_nodes)
            coobserved = new_node in current_observed_nodes and old_node in current_observed_nodes
            if score >= min_score or coobserved:
                scored.append((score, old_node))
        scored.sort(key=lambda item: item[0], reverse=True)
        max_candidates = int(self.edge_lifecycle_config['max_candidates_per_new_node'])
        return scored[:max_candidates]

    def _refresh_edge_lifecycle(self, edge):
        self._ensure_edge_metadata(edge)
        if edge.status == 'rejected' or edge.relation is None:
            return
        min_evidence = int(self.edge_lifecycle_config['min_active_evidence'])
        min_conf = float(self.edge_lifecycle_config['min_active_confidence'])
        temporal_support = min(1.0, float(edge.n_simultaneous) / max(1, min_evidence))
        support = max(edge.confidence, 0.45 * edge.geometry_score + 0.35 * temporal_support + 0.20 * edge.weight)
        edge.confidence = float(np.clip(support, 0.0, 1.0))
        if edge.evidence_count >= min_evidence and edge.confidence >= min_conf:
            edge.status = 'active'
        else:
            edge.status = 'pending'

    def _initialize_edge_relation(self, edge, relation, geometry_score, current_time):
        relation = self._normalize_relation(relation)
        if relation is None:
            edge.reject()
            edge.delete()
            return False
        edge.geometry_score = geometry_score
        base_conf = float(self.edge_lifecycle_config['initial_relation_confidence'])
        confidence = float(np.clip(base_conf + 0.35 * geometry_score, 0.0, 1.0))
        edge.add_relation_evidence(relation, confidence, source='llm', current_time=current_time)
        self._refresh_edge_lifecycle(edge)
        return True

    def get_scenegraph(self):
        nodes = self._active_nodes()
        edges = self.get_edges()
        caption_count = {}
        
        new_nodes = []
        
        node_id_map = {}
        self.scenegraph_uid_to_node = {}
        for node in nodes:
            caption = node.caption
            if caption not in caption_count:
                caption_count[caption] = 0
            unique_id = "{}_{}".format(caption, caption_count[caption])
            new_nodes.append({
                'id': unique_id,
                'position': node.center if node.center is not None and self._in_bounds_rc(node.center) else [0, 0]
            })
            node_id_map[node] = unique_id
            self.scenegraph_uid_to_node[unique_id] = node
            caption_count[caption] += 1

        new_edges = []
        self.scenegraph_edge_key_to_edge = {}
        for edge in edges:
            self._ensure_edge_metadata(edge)
            if edge.relation is None or edge.status != 'active':
                continue
            if edge.node1 not in node_id_map or edge.node2 not in node_id_map:
                continue
            source_id = node_id_map[edge.node1]
            target_id = node_id_map[edge.node2]
            edge_key = tuple(sorted([source_id, target_id]))
            self.scenegraph_edge_key_to_edge[edge_key] = edge
            new_edges.append({
                'source': source_id,
                'target': target_id,
                'type': edge.relation,
                'weight': edge.weight,
                'confidence': edge.confidence,
                'status': edge.status
            })

        self.scenegraph = {
            'nodes': new_nodes,
            'edges': new_edges
        }

        return self.scenegraph

    def get_nodes(self):
        return self.nodes
    
    def get_edges(self):
        edges = set()
        for node in self.nodes:
            edges.update(node.edges)
        edges = list(edges)
        return edges

    def get_seg_xyxy(self):
        return self.seg_xyxy

    def get_seg_caption(self):
        return self.seg_caption

    def init_room_nodes(self):
        room_nodes = []
        for caption in self.rooms:
            room_node = RoomNode(caption)
            room_nodes.append(room_node)
        self.room_nodes = room_nodes

    def get_grounded_sam(self, device):
        model = load_model(self.groundingdino_config_file, self.groundingdino_checkpoint, self.bert_base_uncased_path, device=device)
        predictor = SamPredictor(sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(device))
        return model, predictor
    
    def get_segmentation(
        self, model, image: np.ndarray
    ) -> tuple:
        groundingdino = model[0]
        sam_predictor = model[1]
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_resized, _ = transform(Image.fromarray(image), None)  # 3, h, w
        boxes_filt, caption = get_grounding_output(groundingdino, image_resized, caption=self.node_space, box_threshold=0.3, text_threshold=0.25, with_logits=False, device=self.device)
        if len(caption) == 0:
            return None, None, None, None
        sam_predictor.set_image(image)

        # size = image_pil.size
        H, W = image.shape[0], image.shape[1]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        mask, conf, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )
        mask, xyxy, conf = mask.squeeze(1).cpu().numpy(), boxes_filt.squeeze(1).numpy(), conf.squeeze(1).cpu().numpy()
        return mask, xyxy, conf, caption

    def get_pose_matrix(self):
        x = self.map_size_cm / 100.0 / 2.0 + self.observations['gps'][0]
        y = self.map_size_cm / 100.0 / 2.0 - self.observations['gps'][1]
        t = (self.observations['compass'] - np.pi / 2)[0] # input degrees and meters
        yaw_pose = np.array([
            [np.cos(t), -np.sin(t), 0, x],
            [np.sin(t), np.cos(t), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        if not hasattr(self.args, 'camera_tilt_deg') or abs(self.args.camera_tilt_deg) < 1e-8:
            return yaw_pose
        # Navigation point clouds use x=right, y=forward, z=down. Habitat's
        # negative pitch therefore needs the opposite sign to rectify the
        # camera cloud into the level agent frame before applying yaw.
        tilt = np.deg2rad(-float(getattr(self.args, 'camera_tilt_deg', 0.0)))
        tilt_pose = np.array([
            [1, 0, 0, 0],
            [0, np.cos(tilt), -np.sin(tilt), 0],
            [0, np.sin(tilt), np.cos(tilt), 0],
            [0, 0, 0, 1],
        ])
        return yaw_pose @ tilt_pose

    def segment2d(self):
        """对RGB图像进行语义分割"""
        print('    segement2d...')
        with torch.no_grad():
            print('        sam_segmentation...')
            mask, xyxy, masks_conf, caption = self.get_segmentation(self.grounded_sam, self.image_rgb)

            self.seg_xyxy = xyxy
            self.seg_caption = caption
            self.clear_line()
        if caption is None:
            self.clear_line()
            return
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=masks_conf,
            class_id=np.zeros_like(masks_conf).astype(int),
            mask=mask,
        )
        image_appear_efficiency = [''] * len(mask)
        self.segment2d_results.append({
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask,
            "classes": self.classes,
            "image_appear_efficiency": image_appear_efficiency,
            "image_rgb": self.image_rgb,
            "caption": caption,
        })
        self.clear_line()


    def mapping3d(self):
        """将2D检测结果映射到3D空间"""
        print('    mapping3d...')
        depth_array = habitat_depth_to_metric(
            self.image_depth,
            self.args.min_depth,
            self.args.max_depth,
            far_clip_value=0.0,
        )

        gobs = self.segment2d_results[-1]
        
        unt_pose = self.pose_matrix
        
        adjusted_pose = unt_pose
        cam_K = self.camera_matrix
            
        idx = len(self.segment2d_results) - 1
        #将2d检测转换成3d检测表

        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg = self.cfg,
            image = self.image_rgb,
            depth_array = depth_array,
            cam_K = cam_K,
            idx = idx,
            gobs = gobs,
            trans_pose = adjusted_pose,
            class_names = self.classes,
            BG_CLASSES = self.BG_CLASSES,
            is_navigation = self.is_navigation
        )
        
        if len(fg_detection_list) == 0:
            self.clear_line()
            return
            
        if len(self.objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                self.objects.append(fg_detection_list[i])

            # Skip the similarity computation 
            self.objects_post = filter_objects(self.cfg, self.objects)
            self.clear_line()
            return
                
        print('        compute_spatial_similarities...')
        #计算空间相似度
        spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
        self.clear_line()
        spatial_sim[spatial_sim < self.cfg.sim_threshold_spatial] = float('-inf')
        # 合并检测到已有物体或添加为新物体
        self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, spatial_sim)
        
        self.objects_post = filter_objects(self.cfg, self.objects)
        self.clear_line()
            
    def get_caption(self):
        print('    get_caption...')
        for idx, object in enumerate(self.objects_post):
            caption_list = []
            for idx_det in range(len(object["image_idx"])):
                caption = self.segment2d_results[object["image_idx"][idx_det]]['caption'][object["mask_idx"][idx_det]]
                caption_list.append(caption)
            caption = self.find_modes(caption_list)[0]
            object['captions'] = [caption]
        self.clear_line()

    def update_node(self):
        """更新场景图节点（含时序状态更新）"""
        print('    update_node...')
        current_time = self.navigate_steps
        latest_idx = len(self.segment2d_results) - 1 if len(self.segment2d_results) > 0 else -1
        
        # update existing nodes
        for node in self.nodes:
            caption_ori = node.caption
            caption_new = node.object['captions'][0]
            if caption_ori != caption_new:
                node.set_caption(caption_new)
            
            # ★ 核心修复：只对本帧实际观测到的节点更新时序状态
            # UniGoal 会将当前帧的 idx 加入到被观测物体的 image_idx 列表中
            if latest_idx >= 0 and latest_idx in node.object.get('image_idx', []):
                det_confidence = self._node_detection_confidence(node.object, node.score)
                node.update_temporal_state(
                    current_time,
                    similarity=det_confidence,
                    det_confidence=det_confidence,
                )
        
        # add new nodes
        new_objects = list(filter(lambda object: 'node' not in object, self.objects_post))
        for new_object in new_objects:
            new_node = ObjectNode()
            caption = new_object['captions'][0]
            new_node.set_caption(caption)
            new_node.set_object(new_object)
            
            # 初始化新节点的时序状态 (新节点必然在当前帧被观测到)
            det_confidence = self._node_detection_confidence(new_object, 0.5)
            new_node.update_temporal_state(
                current_time,
                similarity=det_confidence,
                det_confidence=det_confidence,
            )
            self.nodes.append(new_node)
        
        # get node.center and node.room
        for node in self.nodes:
            points = np.asarray(node.object['pcd'].points)
            center = points.mean(axis=0)
            map_h, _ = self._full_map_shape()
            col = int(center[0] * 100 / self.map_resolution)
            row = int(center[1] * 100 / self.map_resolution)
            row = map_h - 1 - row
            center_rc = [row, col]
            if not self._in_bounds_rc(center_rc):
                node.set_center(None)
                if node.room_node is not None:
                    node.room_node.nodes.discard(node)
                    node.room_node = None
                continue
            node.set_center(center_rc)
            if hasattr(self, 'room_map'):
                if sum(self.room_map[0, :, row, col]!=0).item() == 0:
                    room_label = 0
                else:
                    room_label = torch.where(self.room_map[0, :, row, col]!=0)[0][0].item()
            else:
                room_label = 0
            if node.room_node is not self.room_nodes[room_label]:
                if node.room_node is not None:
                    node.room_node.nodes.discard(node)
                node.room_node = self.room_nodes[room_label]
                node.room_node.nodes.add(node)
        
        # ===== 计算所有节点的贝叶斯置信度 =====
        self._refresh_node_confidences(current_time)
        
        self.clear_line()

    def create_new_edge(self, new_node):
        # new_edges = []
        for j, old_node in enumerate(self.nodes):
            image = self.get_joint_image(old_node, new_node)
            if image is not None:
                response = self.vlm(self.prompt_create_relation.format(obj1=old_node.caption, obj2=new_node.caption), image)
                if "No clear spatial relationship" not in response:
                    response = response.lower()
                    objects = [old_node.caption.lower(), new_node.caption.lower()]
                    relations = self.graphbuilder.get_relations(response, objects)
                    new_edge = Edge(old_node, new_node)
                    if len(relations) > 0:
                        new_edge.set_relation(relations[0]['type'])
                    else:
                        new_edge.set_relation('')
                    # new_node.edges.add(new_edge)
                    # old_node.edges.add(new_edge)
                    # new_edges.append(new_edge)

    def update_edge(self):
        """更新场景图边,利用LLM推理物体间的空间关系（含时序权重更新）"""
        print('    update_edge...')
        current_time = self.navigate_steps
        
        old_nodes = []
        new_nodes = []
        for i, node in enumerate(self.nodes):
            if node.is_new_node:
                new_nodes.append(node)
                node.is_new_node = False
            else:
                old_nodes.append(node)

        existing_edges = self.get_edges()
        current_observed_nodes = set()

        # ===== 记录共现观测 =====
        if len(self.segment2d_results) > 0:
            current_observed_nodes = set()
            for node in self.nodes:
                if node.t_last == current_time:
                    current_observed_nodes.add(node)
            for edge in existing_edges:
                self._ensure_edge_metadata(edge)
                if edge.node1 in current_observed_nodes and edge.node2 in current_observed_nodes:
                    edge.record_cooccurrence(current_time)
                    self._refresh_edge_lifecycle(edge)

        # ===== 更新现有边的时序权重 =====
        MIN_EDGE_AGE = 20  # 保护期：新边在前 20 步内不会被遗忘阈值删除
        edges_to_remove = []
        for edge in existing_edges:
            self._ensure_edge_metadata(edge)
            weight = edge.update_temporal_weight(current_time)
            edge_age = current_time - edge.t_create
            self._refresh_edge_lifecycle(edge)
            #如果权重低于阈值，标记删除（保护期内跳过）
            if edge.status == 'rejected':
                edges_to_remove.append(edge)
            elif edge_age >= MIN_EDGE_AGE and weight < getattr(self.cfg, 'forget_threshold', 0.1):
                edge.reject()
                edges_to_remove.append(edge)
            elif edge.check_temporal_paradox():
                print(f'        WARNING: Temporal paradox detected for edge ({edge.node1.caption}, {edge.node2.caption})')

        # 执行删除
        for edge in edges_to_remove:
            edge.delete()
        
        if len(new_nodes) == 0:
            self.clear_line()
            return
        
        # create the edge between new_node and old_node (with current_time)
        new_edges = []
        for i, new_node in enumerate(new_nodes):
            for geometry_score, old_node in self._select_edge_candidates(new_node, old_nodes, current_observed_nodes):
                new_edge = Edge(new_node, old_node, current_time)
                new_edge.geometry_score = geometry_score
                new_edges.append(new_edge)
        # create the edge between new_nodes
        for i, new_node1 in enumerate(new_nodes):
            for j, new_node2 in enumerate(new_nodes[i + 1:]):
                if self._find_edge(new_node1, new_node2) is not None:
                    continue
                geometry_score = self._edge_geometry_score(new_node1, new_node2, current_observed_nodes)
                if geometry_score < float(self.edge_lifecycle_config['min_geometry_score']):
                    continue
                new_edge = Edge(new_node1, new_node2, current_time)
                new_edge.geometry_score = geometry_score
                new_edges.append(new_edge)
        # new_edges already contains gated candidate edges.
        # get all relation proposals
        if len(new_edges) > 0:
            print(f'        LLM get all relation proposals...')
            node_pairs = []
            for new_edge in new_edges:
                node_pairs.append(new_edge.node1.caption)
                node_pairs.append(new_edge.node2.caption)
            prompt = self.prompt_edge_proposal + '\n({}, {})' * len(new_edges)
            prompt = prompt.format(*node_pairs)
            relations = self.llm(prompt=prompt)
            relations = [r.strip() for r in relations.split('\n') if r.strip()]
            if len(relations) != len(new_edges):
                print(f'        WARNING: LLM returned {len(relations)} relations for {len(new_edges)} edges, assigning partial')
            for i in range(min(len(relations), len(new_edges))):
                self._initialize_edge_relation(new_edges[i], relations[i], new_edges[i].geometry_score, current_time)
            for i in range(len(relations), len(new_edges)):
                new_edges[i].reject()
                new_edges[i].delete()
            self.clear_line()
        self.clear_line()

    def update_group(self):
        for room_node in self.room_nodes:
            room_node.group_nodes = []
            object_nodes = [
                node for node in room_node.nodes
                if self._node_is_active(node, require_center=True)
            ]
            if len(object_nodes) > 0:
                centers = [object_node.center for object_node in object_nodes]
                centers = np.array(centers)
                dbscan = DBSCAN(eps=10, min_samples=1)  
                clusters = dbscan.fit_predict(centers)  
                for i in range(clusters.max() + 1):
                    group_node = GroupNode()
                    indices = np.where(clusters == i)[0]
                    for index in indices:
                        group_node.nodes.append(object_nodes[index])
                    group_node.get_graph()
                    room_node.group_nodes.append(group_node)
    
    def insert_goal(self, goal=None):
        if goal is None:
            goal = self.obj_goal
        self.update_group()
        room_node_text = ''
        for room_node in self.room_nodes:
            if len(room_node.group_nodes) > 0:
                room_node_text = room_node_text + room_node.caption + ','
        # room_node_text[-2] = '.'
        if room_node_text == '':
            return None
        prompt = self.prompt_room_predict.format(goal, room_node_text)
        response = self.llm(prompt=prompt)
        response = response.lower()
        predict_room_node = None
        for room_node in self.room_nodes:
            if len(room_node.group_nodes) > 0 and room_node.caption.lower() in response:
                predict_room_node = room_node
        if predict_room_node is None:
            return None
        for group_node in predict_room_node.group_nodes:
            corr_score = self.graph_corr(goal, group_node)
            group_node.corr_score = float(np.clip(corr_score, 0.0, 1.0))
        sorted_group_nodes = sorted(predict_room_node.group_nodes)
        self.mid_term_goal = sorted_group_nodes[-1].center
        hint_confidence = float(np.clip(sorted_group_nodes[-1].corr_score, 0.0, 1.0))
        self._set_semantic_goal_state('room_group_hint', hint_confidence, False)
        return self.mid_term_goal
    
    def update_scenegraph(self):
        print(f'update_observation {self.navigate_steps}...')
        self._last_obs_step = self.navigate_steps
        prev_num_segment_results = len(self.segment2d_results)
        self.segment2d()
        has_current_detections = len(self.segment2d_results) > prev_num_segment_results
        if not has_current_detections:
            if not self.disable_temporal:
                self.update_fov_masking()
            self._refresh_node_confidences(self.navigate_steps)
            self.update_edge()
            self._update_current_visit()
            self.get_scenegraph()
            self._update_goal_memory_from_nodes()
            self.clear_line()
            return
        self.mapping3d()
        self.get_caption()
        self.update_node()
        if not self.disable_temporal:
            self.update_fov_masking()
        self.update_edge()

        # [新增] 记录机器人当前位置，用于后续前沿的重访惩罚
        self._update_current_visit()
        
        self.get_scenegraph()
        self._update_goal_memory_from_nodes()
        self.clear_line()

    def explore(self):
        self._reset_semantic_goal_state()
        remembered_goal = self.recall_current_goal_from_memory()
        if remembered_goal is not None:
            return self.get_goal(goal=remembered_goal)
        # STAGE 1: 异常检测优先中断
        is_anomaly, anomaly_goal = self.anomaly_detector()
        if is_anomaly:
            print('[Temporal Graph] 触发主动纠错探测！直接前往异常坐标重验证。')
            self._set_semantic_goal_state('temporal_anomaly', 1.0, True)
            return self.get_goal(goal=anomaly_goal)

        # STAGE 1.5: 门口窥视——到达新区域时 VLM 判断房间类型，跳过不相关房间
        self.doorway_peek()

        # STAGE 2: 超图推理或传统匹配
        if self.hypergraph_config.get('use_hypergraph', True):
            # 使用超图概率推断
            overlap, goal = self.hypergraph_infer()
        else:
            # 使用原有硬匹配
            overlap = self.overlap()
            goal = None
            
        # STAGE 3: 基于匹配度的探索策略选择
        if goal is not None:
            # 超图推理已返回目标
            pass
        elif 0.5 <= overlap < 0.9 and hasattr(self, 'matcher') and len(self.matcher.common_nodes) >= 2:
            goal = self.explore_remaining()
            self._set_semantic_goal_state('relative_position_prediction', overlap, False)
        elif overlap >= 0.9 and hasattr(self, 'matcher') and len(self.matcher.common_nodes) < 2:
            goal = self.reasonableness_correction()
        else:
            goal = self.explore_subgraph()
        
        goal = self.get_goal(goal)
        
        return goal
    
    def hypergraph_infer(self):
        """使用超图推理进行概率匹配
        
        替代原有 overlap() 硬匹配，返回概率分布 prob_map 和等效 overlap 分数。
        核心改进：直接复用 set_image_goal/set_text_goal 已解析的 obj_goal + goalgraph，
        避免在 HyperGraphReasoner 内部重复调用 VLM/LLM 导致解析失败。
        
        Returns:
            overlap_score: 等效重叠分数 [0, 1]
            goal: 高概率区域目标（可为 None）
        """
        # 延迟初始化超图推理器
        if self.hypergraph_reasoner is None:
            self.hypergraph_reasoner = HyperGraphReasoner(
                llm_func=self.llm,
                vlm_func=self.vlm,
                config=self.hypergraph_config
            )
            
        # 导出场景图用于超图推理
        scene_graph = self.export_for_hypergraph()
        
        # 场景图为空时直接回退，避免空因子图推断
        if not scene_graph.get("nodes"):
            self.prob_map = {}
            return 0.0, None
        
        # 获取节点时序置信度（使用 UID）
        node_confidences = {
            uid: node.confidence
            for node, uid in getattr(self, 'node_uid_map', {}).items()
        }
            
        # ── 构建 goal_template：复用已有解析结果，不再重复调用 VLM ──
        goal_template = self._build_goal_template()
        
        try:
            # 执行超图推断（传入预构建的 goal_template，跳过 parse_goal）
            self.prob_map, goal_template = self.hypergraph_reasoner.infer(
                scene_graph=scene_graph,
                goal_input=self.obj_goal,
                goal_type="object",
                node_confidences=node_confidences,
                goal_template=goal_template
            )
        except Exception as e:
            print(f'[HyperGraph] infer() failed ({e}), falling back to explore_subgraph.')
            self.prob_map = {}
            return 0.0, None
        
        # 计算等效 overlap 分数（直接复用 goal_template，无需再次 parse_goal）
        overlap_score = self.hypergraph_reasoner.get_overlap_score(self.prob_map, goal_template)
        
        goal = None
        target = goal_template.get("target", "").lower()
        target_stem = self.hypergraph_reasoner._stem(target)

        def _caption_stems(caption):
            """将 caption 拆成词后分别取词干（处理多词短语如 "wooden bed"）。"""
            return {self.hypergraph_reasoner._stem(w) for w in caption.lower().split()}

        target_stems = _caption_stems(target)

        # ── 方式1：直接目标定位（目标节点的后验概率 > 0.5）──
        # 词级别词干匹配：caption "wooden bed" → stems {"wooden","bed"} → 命中 target "bed"
        # 比原来 `target in node.caption` 精确：不会误匹配子串
        best_node, best_prob = None, 0.0
        for node in getattr(self, 'node_uid_map', {}).keys():
            uid = self.node_uid_map.get(node, None)
            if uid is None:
                continue
            prob = self.prob_map.get(uid, 0.0)
            if self.args.goal_type == 'goat':
                target_matches = self._goal_labels_match(target, node.caption)
            else:
                target_matches = target_stem in _caption_stems(node.caption)
            if target_matches and prob > best_prob:
                best_prob = prob
                best_node = node
        if (
            best_node is not None
            and best_node.center is not None
            and self._in_bounds_rc(best_node.center)
            and best_prob > 0.5
        ):
            goal = best_node.center
            self._set_semantic_goal_state('hypergraph_direct_target', best_prob, True)
            print(f'[HyperGraph] Direct target: "{best_node.caption}" (p={best_prob:.2f}) → {goal}')
            return overlap_score, goal

        # ── 方式2：上下文引导定位（目标未见，但≥2个高置信度上下文节点可三角定位）──
        # 当场景中已有与目标强关联的物体（如 nightstand/lamp 暗示 bed 在附近），
        # 用这些锚点的加权质心预测目标位置，比纯LLM预测更精准
        context_stem_sets = [
            _caption_stems(context) for context in goal_template.get("context", [])
        ]
        high_conf_ctx = []
        for node in getattr(self, 'node_uid_map', {}).keys():
            uid = self.node_uid_map.get(node, None)
            if uid is None or node.center is None or not self._in_bounds_rc(node.center):
                continue
            prob = self.prob_map.get(uid, 0.0)
            caption_stems = _caption_stems(node.caption)
            if self.args.goal_type == 'goat':
                context_matches = any(stems <= caption_stems for stems in context_stem_sets)
            else:
                legacy_context_stems = {
                    self.hypergraph_reasoner._stem(context)
                    for context in goal_template.get("context", [])
                }
                context_matches = bool(caption_stems & legacy_context_stems)
            if context_matches and prob > 0.6:
                high_conf_ctx.append((node, prob))

        if len(high_conf_ctx) >= 2:
            total_w = sum(p for _, p in high_conf_ctx)
            cx = sum(n.center[0] * p for n, p in high_conf_ctx) / total_w
            cy = sum(n.center[1] * p for n, p in high_conf_ctx) / total_w
            candidate_goal = [cx, cy]
            goal = candidate_goal if self._in_bounds_rc(candidate_goal) else None
            if goal is not None:
                self._set_semantic_goal_state(
                    'hypergraph_context_triangulation',
                    max(p for _, p in high_conf_ctx),
                    False,
                )
            ctx_names = [n.caption for n, _ in high_conf_ctx[:3]]
            print(f'[HyperGraph] Context triangulation: {ctx_names} → ({cx:.0f}, {cy:.0f})')

        return overlap_score, goal
    
    def _build_goal_template(self):
        """从已有的 obj_goal + goalgraph 构建超图推理所需的 goal_template。
        
        set_image_goal() 流程中，VLM 已将图像描述转为 text_goal，
        GraphBuilder 又从 text_goal 提取了结构化 goalgraph（节点+边）。
        此方法直接复用这些成果，无需再次调用 VLM/LLM。
        
        Returns:
            goal_template: {"target": str, "context": [str], "constraints": [str]}
        """
        target = str(self.obj_goal).lower().strip() if self.obj_goal else "unknown"
        
        # 从 goalgraph 提取上下文物体（排除目标本身）
        context = []
        constraints = []
        if hasattr(self, 'goalgraph') and self.goalgraph is not None:
            for node in self.goalgraph.get('nodes', []):
                obj_name = node.get('id', '').lower().strip()
                if obj_name and obj_name != target:
                    context.append(obj_name)
            # 从 goalgraph 边提取空间约束
            for edge in self.goalgraph.get('edges', []):
                src = edge.get('source', '')
                tgt = edge.get('target', '')
                rel = edge.get('type', '')
                if src and tgt and rel:
                    constraints.append(f"{src} is {rel} {tgt}")
        
        # 如果 goalgraph 没有提供足够上下文，用 LLM 扩展
        if len(context) < 2 and target != "unknown":
            expanded = self.hypergraph_reasoner._expand_goal_context(target)
            # 合并去重
            existing = set(context)
            for c in expanded:
                if c not in existing and c != target:
                    context.append(c)
                    existing.add(c)

        if (
            self.hypergraph_reasoner is not None
            and hasattr(self.hypergraph_reasoner, '_filter_context_candidates')
        ):
            context = self.hypergraph_reasoner._filter_context_candidates(
                context, target=target, limit=5
            )
        
        return {
            "target": target,
            "context": context[:5],
            "constraints": constraints
        }
    
    def doorway_peek(self):
        """门口窥视：在前沿位置利用 VLM 判断前方房间类型，过滤不相关区域。
        
        当机器人到达前沿（未知区域边界）时调用。通过当前 RGB 图像
        调用 VLM 识别前方房间类型，再用 LLM 判断该房间是否可能包含
        目标物体。若不相关，则记录该位置，后续前沿评分会惩罚附近区域。
        
        Returns:
            room_type: 识别到的房间类型（str），若未触发窥视则返回 None
        """
        if not self.peek_config.get('enabled', True):
            return None
        if not hasattr(self, 'image_rgb') or self.image_rgb is None:
            return None
            
        # 计算当前机器人地图坐标
        map_h, map_w = self._full_map_shape()
        current_col = map_w / 2.0 + (self.observations['gps'][0] * 100.0 / self.map_resolution)
        current_row_world = map_h / 2.0 - (self.observations['gps'][1] * 100.0 / self.map_resolution)
        current_row = map_h - 1 - current_row_world
        current_pos = np.array([current_row, current_col])
        
        # 节流：与上次窥视位置距离不足则跳过
        min_dist = self.peek_config.get('min_move_dist', 30)
        if self._peek_last_pos is not None:
            if np.linalg.norm(current_pos - self._peek_last_pos) < min_dist:
                return None
        
        # 调用 VLM 识别房间类型
        try:
            image = Image.fromarray(self.image_rgb) if isinstance(self.image_rgb, np.ndarray) else self.image_rgb
            raw_room_type = self.vlm(self.prompt_room_classify, image)
            room_type = self._normalize_room_type(raw_room_type)
        except Exception as e:
            print(f'        [DoorwayPeek] VLM room classification failed: {e}')
            return None
        
        self._peek_last_pos = current_pos.copy()
        
        # 判断房间与目标的相关性
        target = str(self.obj_goal).lower().strip() if self.obj_goal else ""
        is_relevant = self._is_room_relevant(room_type, target)
        
        if is_relevant:
            print(f'        [DoorwayPeek] Room "{room_type}" is RELEVANT to "{target}" → proceed exploring')
        else:
            print(f'        [DoorwayPeek] Room "{room_type}" is IRRELEVANT to "{target}" → skip region')
            if room_type != 'unknown':
                self._peeked_irrelevant_regions.append(
                    (current_pos[0], current_pos[1], room_type)
                )
        
        return room_type
    
    def _is_room_relevant(self, room_type, target):
        """判断房间类型是否可能包含目标物体。
        
        先查缓存，未命中则调用 LLM 判断。
        
        Args:
            room_type: 房间类型（如 "bedroom", "kitchen"）
            target: 目标物体名（如 "chair"）
            
        Returns:
            bool: True 表示相关，值得探索
        """
        if not room_type or not target:
            return True  # 信息不足时默认相关，避免误跳过
        room_type = self._normalize_room_type(room_type)
        if room_type == 'unknown':
            return True
        
        cache_key = (room_type, target)
        if cache_key in self._room_relevance_cache:
            return self._room_relevance_cache[cache_key]
        
        # 走廊/过道始终相关（通道不应被跳过）
        passthrough_types = getattr(self, 'passthrough_room_types', {'hallway', 'corridor', 'passage', 'entryway', 'foyer', 'lobby'})
        if room_type in passthrough_types or any(p in room_type for p in passthrough_types):
            self._room_relevance_cache[cache_key] = True
            return True

        prior_relevance = self._target_room_prior_relevance(room_type, target)
        if prior_relevance is not None:
            self._room_relevance_cache[cache_key] = prior_relevance
            return prior_relevance
        
        try:
            prompt = self.prompt_room_relevance.format(room_type=room_type, target=target)
            response = self.llm(prompt=prompt).strip().lower()
            words = response.split()
            first_word = words[0] if words else ''
            is_relevant = first_word.startswith('yes')
        except Exception as e:
            print(f'        [DoorwayPeek] LLM relevance check failed: {e}')
            is_relevant = True  # 失败时默认相关
        
        self._room_relevance_cache[cache_key] = is_relevant
        return is_relevant
    
    def _compute_peek_frontier_penalty(self, frontier_locations):
        """计算前沿点的门口窥视惩罚分数。
        
        对于靠近"已标记为不相关"区域的前沿，施加负分惩罚，
        引导机器人远离不相关房间。
        
        Args:
            frontier_locations: 前沿位置数组 (N, 2)
            
        Returns:
            penalty_scores: 惩罚分数数组 (N,)，全 ≤ 0
        """
        num_frontiers = len(frontier_locations)
        penalty_scores = np.zeros(num_frontiers)
        
        if not self._peeked_irrelevant_regions:
            return penalty_scores
        
        penalty_weight = self.peek_config.get('irrelevant_penalty', 5.0)
        penalty_radius = self.peek_config.get('penalty_radius', 80)
        
        for (rx, ry, _room_type) in self._peeked_irrelevant_regions:
            region_pos = np.array([rx, ry])
            for i, frontier in enumerate(frontier_locations):
                dist = np.linalg.norm(np.array(frontier) - region_pos)
                if dist < penalty_radius:
                    # 距离越近惩罚越大，线性衰减
                    penalty = -penalty_weight * (1.0 - dist / penalty_radius)
                    penalty_scores[i] += penalty
        
        return penalty_scores

    def _compute_arrival_reject_frontier_penalty(self, frontier_locations):
        """Penalize frontiers near locations rejected by the arrival verifier."""
        cfg = self.frontier_config
        return compute_arrival_reject_penalty(
            frontier_locations,
            getattr(self, 'rejected_goal_regions', []),
            self.map_resolution,
            penalty_weight=float(cfg.get('arrival_reject_penalty', 3.0)),
            radius_m=float(cfg.get('arrival_reject_radius_m', 2.0)),
            enabled=bool(cfg.get('arrival_reject_enabled', True)),
        )

    def _compute_exhausted_frontier_penalty(self, frontier_locations):
        """Penalize frontier regions that recently produced no navigation progress."""
        self._expire_frontier_exhausted_regions()
        regions = [
            item.get('center')
            for item in getattr(self, '_frontier_exhausted_regions', [])
            if item.get('center') is not None
        ]
        cfg = self.frontier_config
        return compute_arrival_reject_penalty(
            frontier_locations,
            regions,
            self.map_resolution,
            penalty_weight=float(cfg.get('exhausted_frontier_penalty', 6.0)),
            radius_m=float(cfg.get('exhausted_frontier_radius_m', 2.0)),
            enabled=True,
        )

    def export_for_hypergraph(self):
        """导出场景图结构用于超图推理（带唯一 ID）"""
        self.node_uid_map = {}  # 建立 Node对象 -> UID 的全局映射
        nodes = []
        active_nodes = self._active_nodes()
        active_node_set = set(active_nodes)
        for idx, node in enumerate(active_nodes):
            uid = f"{node.caption}_{id(node)}"  # 利用内存地址生成绝对唯一 ID
            self.node_uid_map[node] = uid
            nodes.append({
                "id": uid,
                "caption": node.caption,
                "position": node.center if node.center is not None and self._in_bounds_rc(node.center) else [0, 0],
                "confidence": node.confidence,
                "c_trend": node.c_trend,
                "n_obs": node.n_obs
            })
            
        edges = []
        seen_edges = set()
        for node in active_nodes:
            for edge in node.edges:
                # 确保无向边只添加一次
                edge_key = tuple(sorted([id(edge.node1), id(edge.node2)]))
                self._ensure_edge_metadata(edge)
                if (
                    edge_key not in seen_edges
                    and edge.node1 in active_node_set
                    and edge.node2 in active_node_set
                    and edge.relation is not None
                    and edge.status != 'rejected'
                ):
                    seen_edges.add(edge_key)
                    edges.append({
                        "source": self.node_uid_map[edge.node1],
                        "target": self.node_uid_map[edge.node2],
                        "source_caption": edge.node1.caption,
                        "target_caption": edge.node2.caption,
                        "type": edge.relation,
                        "weight": edge.weight,
                        "confidence": edge.confidence,
                        "status": edge.status,
                        "evidence_count": edge.evidence_count,
                        "geometry_score": edge.geometry_score
                    })
        return {"nodes": nodes, "edges": edges}
                    

    def explore_subgraph(self, goal=None):
        if goal is None:
            if not self.goalgraph_decomposed or 'subgraph_1' not in self.goalgraph_decomposed:
                return self.insert_goal()
            self.subgraph = self.goalgraph_decomposed['subgraph_1']

        self.subgraph = self.goalgraphdecomposer.graph_to_text(self.subgraph)
        return self.insert_goal(self.subgraph)

    def _build_room_hint_from_prob_map(self):
        """从 prob_map 找后验概率最高的节点所在 room，返回该 room 中最佳 group_node 中心。

        用于当超图推理无法直接定位目标时（prob_map 有信号但目标节点未达阈值），
        通过已有高置信度上下文节点的房间归属，代替纯 LLM 文字预测选房间。

        Returns:
            center: [x, y] 或 None
        """
        active_nodes = self._active_nodes(require_center=True)
        if not self.prob_map or not active_nodes:
            return None

        uid_map = getattr(self, 'node_uid_map', {})
        if not uid_map:
            return None

        best_node, best_prob = None, 0.0
        for node in active_nodes:
            uid = uid_map.get(node)
            if uid is None or node.center is None or not self._in_bounds_rc(node.center):
                continue
            prob = self.prob_map.get(uid, 0.0)
            if prob > best_prob:
                best_prob = prob
                best_node = node

        if best_node is None or best_prob < 0.5:
            return None

        # 找到该节点所属的 room_node
        target_room = None
        for room_node in self.room_nodes:
            if best_node in room_node.nodes:
                target_room = room_node
                break

        if target_room is None:
            return None

        # 确保 group_nodes 已更新
        self.update_group()
        if not target_room.group_nodes:
            return None

        # 选择 corr_score 最高的 group（若已有评分），否则取最大的 group
        best_group = max(
            target_room.group_nodes,
            key=lambda g: getattr(g, 'corr_score', len(g.nodes))
        )
        self.mid_term_goal = best_group.center
        print(f'[HyperGraph→Room] "{best_node.caption}" (p={best_prob:.2f}) '
              f'→ room "{target_room.caption}" → group center {self.mid_term_goal}')
        return self.mid_term_goal
    
    def overlap(self):
        graph1 = self.scenegraph
        graph2 = self.goalgraph
        self.matcher = GraphMatcher(graph1, graph2, self.llm)
        overlap_score = self.matcher.overlap()
        return overlap_score
    
    def explore_remaining(self):
        G1 = self.matcher.G1
        G2 = self.matcher.G2
        common_nodes = self.matcher.common_nodes

        # Assign positions to the first two common nodes in the subgraph
        for i, node_id in enumerate(common_nodes):
            if i < 2:
                G2.nodes[node_id]['position'] = G1.nodes[node_id]['position']
            else:
                break

        # Calculate relative positions within the subgraph
        positions = self.matcher.calculate_relative_positions(G2, common_nodes)

        # Predict positions of the remaining nodes
        position = self.matcher.predict_remaining_node_positions(common_nodes, positions, G1)
        return position

    def _apply_corrected_scenegraph(self, corrected_graph):
        if not hasattr(self, 'scenegraph_edge_key_to_edge'):
            return
        corrected_edges = {}
        for edge_data in corrected_graph.get('edges', []):
            source = edge_data.get('source')
            target = edge_data.get('target')
            relation = self._normalize_relation(edge_data.get('type'))
            if source is None or target is None or relation is None:
                continue
            corrected_edges[tuple(sorted([source, target]))] = relation

        for edge_key, edge in list(self.scenegraph_edge_key_to_edge.items()):
            self._ensure_edge_metadata(edge)
            if edge_key not in corrected_edges:
                edge.reject()
                edge.delete()
                continue
            corrected_relation = corrected_edges[edge_key]
            if edge.relation != corrected_relation:
                edge.set_relation(corrected_relation)
                edge.confidence = float(np.clip(edge.confidence * 0.7, 0.0, 1.0))
                edge.status = 'pending'
            self._refresh_edge_lifecycle(edge)

    def reasonableness_correction(self):
        corrector = SceneGraphCorrector(self.llm)
        if not hasattr(self, 'scenegraph'):
            self.get_scenegraph()
        start_node = getattr(self, 'obj_goal', None) or self.text_goal or ''
        scene_node_ids = {node.get('id') for node in self.scenegraph.get('nodes', [])}
        if start_node not in scene_node_ids:
            for node_id in scene_node_ids:
                if str(node_id).lower().startswith(str(start_node).lower()):
                    start_node = node_id
                    break
        corrected_graph = corrector.correct_scene_graph(self.scenegraph, start_node)
        self._apply_corrected_scenegraph(corrected_graph)
        self.scenegraph = corrected_graph
        return None

    def clear_line(self, line_num=1):
        for i in range(line_num):  
            sys.stdout.write('\033[F')
            sys.stdout.write('\033[J')
            sys.stdout.flush()  
    
    def find_modes(self, lst):  
        if len(lst) == 0:
            return ['object']
        else:
            counts = Counter(lst)  
            max_count = max(counts.values())  
            modes = [item for item, count in counts.items() if count == max_count]  
            return modes  
        
    def get_joint_image(self, node1, node2):
        image_idx1 = node1.object["image_idx"]
        image_idx2 = node2.object["image_idx"]
        image_idx = set(image_idx1) & set(image_idx2)
        if len(image_idx) == 0:
            return None
        conf_max = -np.inf
        # get joint images of the two nodes
        for idx in image_idx:
            conf1 = node1.object["conf"][image_idx1.index(idx)]
            conf2 = node2.object["conf"][image_idx2.index(idx)]
            conf = conf1 + conf2
            if conf > conf_max:
                conf_max = conf
                idx_max = idx
        image = self.segment2d_results[idx_max]["image_rgb"]
        image = Image.fromarray(image)
        return image

    def get_goal(self, goal=None):
        if (
            self.args.goal_type == 'goat'
            and isinstance(goal, (list, tuple, np.ndarray))
            and bool(getattr(self, '_semantic_direct_allowed', False))
            and getattr(self, '_semantic_goal_source', '') in (
                'goat_memory',
                'hypergraph_direct_target',
            )
        ):
            direct_rc = self._to_rc(goal)
            if self._in_bounds_rc(direct_rc):
                self._semantic_goal_selected_direct = True
                print(
                    f"[GoalCheck] Direct GOAT semantic recall: "
                    f"goal={direct_rc}, source={self._semantic_goal_source}, "
                    f"conf={self._semantic_goal_confidence:.2f}"
                )
                return direct_rc

        fbe_map = torch.zeros_like(self.full_map[0,0])
        if self.full_map.shape[1] == 1:
            fbe_map[self.fbe_free_map[0,0]>0] = 1
        else:
            fbe_map[self.full_map[0,1]>0] = 1
        fbe_map[skimage.morphology.binary_dilation(self.full_map[0,0].cpu().numpy(), skimage.morphology.disk(4))] = 3 # then dialte obstacle

        fbe_cp = copy.deepcopy(fbe_map)
        fbe_cpp = copy.deepcopy(fbe_map)
        fbe_cp[fbe_cp==0] = 4 # don't know space is 4
        fbe_cp[fbe_cp<4] = 0 # free and obstacle
        selem = skimage.morphology.disk(1)
        fbe_cpp[skimage.morphology.binary_dilation(fbe_cp.cpu().numpy(), selem)] = 0 # don't know space is 0 dialate unknown space
        
        diff = fbe_map - fbe_cpp # intersection between unknown area and free area 
        frontier_map = diff == 1
        frontier_map = remove_small_frontiers(frontier_map, min_size=20)
        frontier_locations = torch.stack([torch.where(frontier_map)[0], torch.where(frontier_map)[1]]).T
        num_frontiers = len(torch.where(frontier_map)[0])
        if num_frontiers == 0:
            return self._empty_frontier_result('no_frontiers_after_morphology', num_raw=0)
        
        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy()
        input_pose[1] = self.map_size_cm/100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        traversible, start = self.get_traversible(self.full_map.cpu().numpy()[0, 0, ::-1], input_pose)
        planner = FMMPlanner(traversible)
        state = [start[0] + 1, start[1] + 1]
        planner.set_goal(state)
        fmm_dist = planner.fmm_dist[::-1]
        self._update_frontier_progress(fmm_dist)

        frontier_locations = (frontier_locations + 1).cpu().numpy()
        in_bounds = (
            (frontier_locations[:, 0] >= 0)
            & (frontier_locations[:, 0] < fmm_dist.shape[0])
            & (frontier_locations[:, 1] >= 0)
            & (frontier_locations[:, 1] < fmm_dist.shape[1])
        )
        frontier_locations = frontier_locations[in_bounds]
        if len(frontier_locations) == 0:
            return self._empty_frontier_result(
                'frontiers_outside_fmm_bounds',
                num_raw=num_frontiers,
                fmm_shape=list(fmm_dist.shape),
            )

        frontier_goals = frontier_locations - 1
        raw_fmm_distances = fmm_dist[frontier_locations[:, 0], frontier_locations[:, 1]]
        distances = raw_fmm_distances / 20.0
        traversible_for_frontiers = traversible[::-1]
        traversible_ok = traversible_for_frontiers[
            frontier_locations[:, 0], frontier_locations[:, 1]
        ] > 0

        cfg = self.frontier_config
        finite_fmm = np.isfinite(fmm_dist)
        if not np.any(finite_fmm):
            return self._empty_frontier_result(
                'invalid_fmm_distance_field',
                num_raw=len(frontier_goals),
                num_in_bounds=len(frontier_locations),
            )
        max_fmm = float(np.nanmax(fmm_dist[finite_fmm]))
        fmm_reachable = (
            np.isfinite(distances)
            & (raw_fmm_distances < max_fmm - 1e-5)
        )
        fmm_ceiling_fallback = False
        finite_reachable = fmm_reachable
        if not np.any(finite_reachable):
            finite_reachable = np.isfinite(distances) & traversible_ok
            fmm_ceiling_fallback = bool(np.any(finite_reachable))
        strict_reachable = (
            finite_reachable
            & (distances >= float(cfg['min_distance_m']))
            & (distances <= float(cfg['max_distance_m']))
            & traversible_ok
        )
        cooldown_mask = self._frontier_cooldown_mask(frontier_goals)
        empty_streak_before = int(getattr(self, '_empty_frontier_streak', 0))
        rescue_mode = False
        candidate_mask = strict_reachable & (~cooldown_mask)
        selection_mode = 'strict_fmm_ceiling_fallback' if fmm_ceiling_fallback else 'strict'
        if not np.any(candidate_mask):
            candidate_mask = strict_reachable
            selection_mode = 'strict_ignore_cooldown'
        if not np.any(candidate_mask):
            candidate_mask = finite_reachable & traversible_ok & (~cooldown_mask)
            selection_mode = 'reachable_traversible_no_distance_gate'
        if not np.any(candidate_mask):
            candidate_mask = finite_reachable & traversible_ok
            selection_mode = 'reachable_traversible_ignore_cooldown'
        if not np.any(candidate_mask):
            candidate_mask = finite_reachable & (~cooldown_mask)
            selection_mode = 'reachable_boundary_no_traversible_gate'
        if not np.any(candidate_mask):
            candidate_mask = finite_reachable
            selection_mode = 'reachable_boundary_ignore_cooldown'
        if (
            not np.any(candidate_mask)
            and empty_streak_before >= int(cfg.get('rescue_patience', 3))
        ):
            rescue_min_m = float(cfg.get('rescue_min_distance_m', 0.3))
            candidate_mask = np.isfinite(distances) & (distances >= rescue_min_m)
            selection_mode = 'rescue_any_finite_frontier'
        if not np.any(candidate_mask):
            return self._empty_frontier_result(
                'no_fmm_reachable_frontiers',
                num_raw=len(frontier_goals),
                num_in_bounds=len(frontier_locations),
                num_fmm_reachable=int(np.sum(fmm_reachable)),
                num_traversible=int(np.sum(traversible_ok)),
                num_strict=int(np.sum(strict_reachable)),
                fmm_max=max_fmm,
                frontier_dist_min_m=float(np.nanmin(distances)) if len(distances) else None,
                frontier_dist_max_m=float(np.nanmax(distances)) if len(distances) else None,
            )
        if fmm_ceiling_fallback and 'fmm_ceiling_fallback' not in selection_mode:
            selection_mode = f'{selection_mode}_fmm_ceiling_fallback'
        rescue_mode = (
            selection_mode.startswith('rescue')
            or 'ignore_cooldown' in selection_mode
            or 'no_distance_gate' in selection_mode
            or 'no_traversible_gate' in selection_mode
            or empty_streak_before >= int(cfg.get('rescue_patience', 3))
        )
        if rescue_mode:
            self._frontier_rescue_count = getattr(self, '_frontier_rescue_count', 0) + 1
        self._empty_frontier_streak = 0

        frontier_locations_16 = frontier_locations[candidate_mask]
        frontier_goals_16 = frontier_goals[candidate_mask]
        distances_16 = distances[candidate_mask]
        cooldown_16 = cooldown_mask[candidate_mask]

        self.frontier_locations = frontier_goals
        self.frontier_locations_16 = frontier_goals_16
        scores = np.zeros(len(frontier_locations_16), dtype=np.float32)
        score_terms = {
            'distance': np.zeros(len(frontier_locations_16), dtype=np.float32),
            'cooldown': np.zeros(len(frontier_locations_16), dtype=np.float32),
            'temporal': np.zeros(len(frontier_locations_16), dtype=np.float32),
            'hypergraph': np.zeros(len(frontier_locations_16), dtype=np.float32),
            'peek': np.zeros(len(frontier_locations_16), dtype=np.float32),
            'arrival_reject': np.zeros(len(frontier_locations_16), dtype=np.float32),
            'exhausted': np.zeros(len(frontier_locations_16), dtype=np.float32),
            'semantic_goal': np.zeros(len(frontier_locations_16), dtype=np.float32),
            'belief': np.zeros(len(frontier_locations_16), dtype=np.float32),
        }

        score_min_m = float(cfg['min_distance_m'])
        score_max_m = float(cfg['max_distance_m'])
        finite_distances_16 = distances_16[np.isfinite(distances_16)]
        if len(finite_distances_16) > 0:
            score_min_m = min(score_min_m, float(np.nanmin(finite_distances_16)))
            score_max_m = max(score_max_m, float(np.nanmax(finite_distances_16)))

        distance_scores = (
            self._normalize_inverse_distance(
                distances_16,
                min_m=score_min_m,
                max_m=score_max_m,
            )
            * float(cfg['distance_weight'])
        )
        score_terms['distance'] = distance_scores.astype(np.float32)
        scores += distance_scores
        if np.any(cooldown_16):
            cooldown_penalty = float(cfg['cooldown_penalty'])
            if rescue_mode:
                cooldown_penalty *= 0.25
            score_terms['cooldown'][cooldown_16] = -cooldown_penalty
            scores += score_terms['cooldown']
        
        # ===== 时序增强的前沿选择 =====
        # 如果启用超图，时序评分只负责“重访惩罚（斥力）”，关闭通用物品吸引力
        if self.hypergraph_config.get('use_hypergraph', True) and self.prob_map:
            temporal_scores = self.compute_temporal_frontier_scores(
                frontier_goals_16, current_time=self.navigate_steps,
                conf_weight=0.0,  # 禁用通用吸引力
                novelty_weight=self.temporal_config['novelty_weight']
            )
        else:
            # 否则，时序评分同时负责吸引和排斥
            temporal_scores = self.compute_temporal_frontier_scores(
                frontier_goals_16, current_time=self.navigate_steps,
                conf_weight=self.temporal_config['conf_weight'],
                novelty_weight=self.temporal_config['novelty_weight']
            )
        score_terms['temporal'] = temporal_scores.astype(np.float32)
        scores += temporal_scores
        
        # ===== 超图概率增强的前沿选择 =====
        if self.hypergraph_config.get('use_hypergraph', True) and self.prob_map:
            hypergraph_scores = self.compute_hypergraph_frontier_scores(
                frontier_goals_16, self.prob_map
            )
            score_terms['hypergraph'] = hypergraph_scores.astype(np.float32)
            scores += hypergraph_scores
        
        # ===== 门口窥视惩罚：远离已确认不相关的房间区域 =====
        peek_penalty_applied = False
        if self.peek_config.get('enabled', True) and self._peeked_irrelevant_regions:
            peek_penalty = self._compute_peek_frontier_penalty(frontier_goals_16)
            peek_penalty = np.maximum(peek_penalty, -float(cfg.get('peek_penalty_limit', 2.5)))
            if not rescue_mode:
                score_terms['peek'] = peek_penalty.astype(np.float32)
                scores += peek_penalty
                peek_penalty_applied = True

        arrival_reject_penalty_applied = False
        if (
            cfg.get('arrival_reject_enabled', True)
            and getattr(self, 'rejected_goal_regions', None)
        ):
            arrival_reject_penalty = self._compute_arrival_reject_frontier_penalty(
                frontier_goals_16
            )
            score_terms['arrival_reject'] = arrival_reject_penalty.astype(np.float32)
            scores += arrival_reject_penalty
            arrival_reject_penalty_applied = bool(np.any(arrival_reject_penalty < 0))

        exhausted_penalty = self._compute_exhausted_frontier_penalty(frontier_goals_16)
        exhausted_penalty_applied = bool(np.any(exhausted_penalty < 0))
        if exhausted_penalty_applied:
            score_terms['exhausted'] = exhausted_penalty.astype(np.float32)
            scores += exhausted_penalty
        
        if isinstance(goal, list) or isinstance(goal, np.ndarray):
            goal_rc = self._to_rc(goal)
            goal_status = self._classify_global_goal(goal_rc)
            direct_goal, direct_dist_m = self._snap_goal_to_reachable(
                goal_rc, traversible, planner.fmm_dist
            )
            goal_source = getattr(self, '_semantic_goal_source', 'unknown')
            goal_confidence = float(getattr(self, '_semantic_goal_confidence', 0.0))
            direct_allowed = bool(getattr(self, '_semantic_direct_allowed', False))
            min_semantic_conf = float(cfg.get('semantic_goal_min_conf', 0.15))
            semantic_bias_allowed = direct_allowed or goal_confidence >= min_semantic_conf
            direct_radius_m = float(cfg['direct_goal_radius_m'])
            if self.args.goal_type == 'goat' and goal_source in (
                'goat_memory',
                'hypergraph_direct_target',
            ):
                direct_radius_m = float(cfg.get('memory_direct_goal_radius_m', direct_radius_m))
            snap_offset_m = None
            if direct_goal is not None and goal_rc is not None:
                snap_offset_m = (
                    float(np.linalg.norm(np.asarray(goal_rc) - np.asarray(direct_goal)))
                    * float(self.map_resolution) / 100.0
                )
            if (
                direct_allowed
                and direct_goal is not None
                and goal_status in ('in_map_explored_free', 'in_map_obstacle')
                and direct_dist_m is not None
                and direct_dist_m <= direct_radius_m
                and (
                    goal_status != 'in_map_obstacle'
                    or (
                        snap_offset_m is not None
                        and snap_offset_m <= float(cfg['direct_goal_snap_max_m'])
                    )
                )
            ):
                self._last_frontier_goal = None
                self._last_frontier_dist_m = None
                self._frontier_stall_count = 0
                print(f'[GoalCheck] Navigate to semantic goal: goal={goal_rc}, snapped={direct_goal}, '
                      f'status={goal_status}, dist={direct_dist_m:.2f}m, '
                      f'source={goal_source}, conf={goal_confidence:.2f}, '
                      f'snap_offset={snap_offset_m:.2f}m')
                self._semantic_goal_selected_direct = True
                return direct_goal
            if not semantic_bias_allowed:
                print(f'[GoalCheck] Skip low-confidence semantic goal bias: goal={goal_rc}, '
                      f'status={goal_status}, source={goal_source}, conf={goal_confidence:.2f}, '
                      f'min_conf={min_semantic_conf:.2f}')
            elif direct_goal is not None and not direct_allowed:
                print(f'[GoalCheck] Use semantic goal as frontier bias only: goal={goal_rc}, '
                      f'status={goal_status}, dist={direct_dist_m}, source={goal_source}, '
                      f'conf={goal_confidence:.2f}')
            elif direct_goal is not None and goal_status == 'in_map_obstacle':
                print(f'[GoalCheck] Skip obstacle semantic direct goal: goal={goal_rc}, '
                      f'snapped={direct_goal}, snap_offset={snap_offset_m}, '
                      f'source={goal_source}, conf={goal_confidence:.2f}')

            state = self._global_rc_to_flipped_fmm_state(goal_rc, traversible.shape)
            if not semantic_bias_allowed:
                pass
            elif state is None or goal_status in ('out_of_global_map', 'invalid'):
                print(f'[GoalCheck] Skip semantic goal bias: goal={goal}, status={goal_status}, '
                      f'full_map={self._full_map_shape()}, traversible={traversible.shape}')
            else:
                planner = FMMPlanner(traversible)
                try:
                    planner.set_goal(state)
                    fmm_dist = planner.fmm_dist[::-1]
                    semantic_distances = fmm_dist[
                        frontier_locations_16[:, 0], frontier_locations_16[:, 1]
                    ] / 20.0
                    semantic_scores = (
                        self._normalize_inverse_distance(semantic_distances)
                        * float(cfg['semantic_goal_weight'])
                        * float(np.clip(goal_confidence if not direct_allowed else 1.0, 0.0, 1.0))
                    )
                    score_terms['semantic_goal'] = semantic_scores.astype(np.float32)
                    scores += semantic_scores
                except ValueError as e:
                    print(f'[GoalCheck] Skip semantic goal bias: {e}')

        belief_info = None
        if cfg.get('belief_enabled', True) and len(frontier_goals_16) > 0:
            belief_info = compute_frontier_belief_utility(
                distances_16,
                positive_evidence={
                    'temporal': score_terms['temporal'],
                    'hypergraph': score_terms['hypergraph'],
                    'semantic_goal': score_terms['semantic_goal'],
                },
                negative_evidence={
                    'peek': score_terms['peek'],
                    'arrival_reject': score_terms['arrival_reject'],
                    'exhausted': score_terms['exhausted'],
                },
                prior=float(cfg.get('belief_prior', 0.35)),
                positive_logit_weight=float(
                    cfg.get('belief_positive_logit_weight', 1.6)
                ),
                negative_logit_weight=float(
                    cfg.get('belief_negative_logit_weight', 1.8)
                ),
                distance_cost_weight=float(
                    cfg.get('belief_distance_cost_weight', 0.25)
                ),
                utility_weight=float(cfg.get('belief_weight', 1.5)),
            )
            score_terms['belief'] = belief_info['utility'].astype(np.float32)
            scores += score_terms['belief']

        best_idx = int(np.argmax(scores))
        goal = frontier_goals_16[best_idx]
        self._last_frontier_goal = goal.tolist()
        self._last_frontier_dist_m = float(distances_16[best_idx])
        self._last_frontier_step = self.navigate_steps
        self.scores = scores
        topk = min(int(cfg['debug_topk']), len(scores))
        order = np.argsort(scores)[-topk:][::-1]
        def _term_breakdown(idx):
            return {
                name: round(float(values[idx]), 3)
                for name, values in score_terms.items()
                if abs(float(values[idx])) > 1e-4
            }

        summary = [
            {
                'goal': frontier_goals_16[i].astype(int).tolist(),
                'dist_m': round(float(distances_16[i]), 2),
                'score': round(float(scores[i]), 3),
                'terms': _term_breakdown(i),
                'nearby_nodes': self._frontier_node_evidence(frontier_goals_16[i]),
            }
            for i in order
        ]
        self.frontier_candidates_debug = summary
        self.frontier_debug = {
            'selected': goal.astype(int).tolist(),
            'selected_dist_m': round(float(distances_16[best_idx]), 2),
            'selected_terms': _term_breakdown(best_idx),
            'selected_nearby_nodes': self._frontier_node_evidence(frontier_goals_16[best_idx]),
            'top': summary,
            'num_raw': int(len(frontier_goals)),
            'num_in_bounds': int(len(frontier_locations)),
            'num_fmm_reachable': int(np.sum(fmm_reachable)),
            'num_reachable_after_fallback': int(np.sum(finite_reachable)),
            'num_traversible': int(np.sum(traversible_ok)),
            'num_strict': int(np.sum(strict_reachable)),
            'num_candidates': int(len(frontier_goals_16)),
            'selection_mode': selection_mode,
            'fmm_ceiling_fallback': fmm_ceiling_fallback,
            'rescue_mode': bool(rescue_mode),
            'rescue_count': int(getattr(self, '_frontier_rescue_count', 0)),
            'empty_frontier_streak_before': int(empty_streak_before),
            'cooldown_count': int(len(getattr(self, '_frontier_cooldowns', []))),
            'cooldown_candidates': int(np.sum(cooldown_16)),
            'peek_penalty_applied': bool(peek_penalty_applied),
            'arrival_reject_penalty_applied': bool(arrival_reject_penalty_applied),
            'exhausted_penalty_applied': bool(exhausted_penalty_applied),
            'exhausted_region_count': int(
                len(getattr(self, '_frontier_exhausted_regions', []))
            ),
            'exhausted_regions': [
                {
                    'center': list(map(int, item.get('center', []))),
                    'reason': item.get('reason'),
                    'count': int(item.get('count', 1)),
                }
                for item in getattr(self, '_frontier_exhausted_regions', [])
                if item.get('center') is not None
            ],
            'rejected_goal_regions': [
                list(map(int, region)) for region in getattr(self, 'rejected_goal_regions', [])
            ],
        }
        if belief_info is not None:
            self.frontier_debug['belief'] = {
                'selected_belief': round(float(belief_info['belief'][best_idx]), 3),
                'selected_positive': round(float(belief_info['positive'][best_idx]), 3),
                'selected_negative': round(float(belief_info['negative'][best_idx]), 3),
                'selected_distance_cost': round(float(belief_info['distance_cost'][best_idx]), 3),
            }
        if self.navigate_steps % 40 == 0:
            print(f"[Frontier] selected={self.frontier_debug['selected']} "
                  f"dist={self.frontier_debug['selected_dist_m']}m "
                  f"candidates={self.frontier_debug['num_candidates']}/{self.frontier_debug['num_raw']} "
                  f"mode={self.frontier_debug['selection_mode']} "
                  f"top={summary}")
        return goal

    def get_traversible(self, map_pred, pose_pred):
        if isinstance(map_pred, torch.Tensor):
            map_pred = map_pred.cpu().numpy()
        if len(map_pred.shape) == 4:
            map_pred = map_pred[0, 0]
        grid = np.rint(map_pred)

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        r, c = start_y, start_x
        start = [int(r*100/self.map_resolution - gy1),
                 int(c*100/self.map_resolution - gx1)]
        # start = [int(start_x), int(start_y)]
        start = pu.threshold_poses(start, grid.shape)
        visited_local = self.visited[gx1:gx2, gy1:gy2]
        sr1, sr2 = max(0, start[0] - 2), min(visited_local.shape[0], start[0] + 3)
        sc1, sc2 = max(0, start[1] - 2), min(visited_local.shape[1], start[1] + 3)
        visited_local[sr1:sr2, sc1:sc2] = 1
        #Get traversible
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        def delete_boundary(mat):
            new_mat = copy.deepcopy(mat)
            return new_mat[1:-1,1:-1]
        
        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        selem = skimage.morphology.square(1)
        traversible = skimage.morphology.binary_dilation(
                    grid[y1:y2, x1:x2],
                    selem) != True

        if not(traversible[start[0], start[1]]):
            print("Not traversible, step is  ", self.navigate_steps)

        # obstacle dilation do not dilate collision
        traversible = 1 - traversible
        selem = skimage.morphology.disk(4)
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem) != True
        
        tr1, tr2 = max(0, int(start[0]-y1)-1), min(traversible.shape[0], int(start[0]-y1)+2)
        tc1, tc2 = max(0, int(start[1]-x1)-1), min(traversible.shape[1], int(start[1]-x1)+2)
        traversible[tr1:tr2, tc1:tc2] = 1
        traversible = traversible * 1.
        
        traversible[self.visited[gx1:gx2, gy1:gy2][y1:y2, x1:x2] == 1] = 1
        traversible = add_boundary(traversible)
        return traversible, start
    
    def reset(self):
        """重置场景图状态（含时序状态）"""
        full_width, full_height = self.map_size, self.map_size
        self.full_width = full_width
        self.full_height = full_height
        self.visited = torch.zeros(full_width, full_height).float().cpu().numpy()
        self.segment2d_results = []
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.group_nodes = []
        self.init_room_nodes()
        self.edge_list = []
        
        # ===== 重置时序相关状态 =====
        self.last_visit_map = np.zeros((full_width, full_height))  # 记录每个位置的最后访问时间
        self.temporal_graph_delta = None  # 图差分缓存
        self.T_forget = 100  # 遗忘阈值（超过此时间未观测的节点标记为消失）
        
        # ===== 重置超图推理状态 =====
        self.prob_map = {}
        self.node_uid_map = {}
        self.scenegraph_uid_to_node = {}
        self.scenegraph_edge_key_to_edge = {}
        self.goal_memory = {}
        if self.hypergraph_reasoner is not None:
            self.hypergraph_reasoner.clear_cache()
        
        # ===== 重置目标图状态 =====
        self.goalgraph = None
        self.goalgraph_decomposed = {}
        self.text_goal = None
        
        # ===== 重置门口窥视状态 =====
        self._peek_last_pos = None
        self._peeked_irrelevant_regions = []
        self._room_relevance_cache = {}
        self._reset_frontier_state()
        self._reset_semantic_goal_state()

    def start_new_goal(self):
        """Reset target-specific reasoning while preserving long-horizon scene memory."""
        self.obj_goal = None
        self.instance_imagegoal = None
        self.text_goal = None
        self.goalgraph = None
        self.goalgraph_decomposed = {}
        self.prob_map = {}
        self.current_goat_goal_type = None
        self._room_relevance_cache = {}
        self._peek_last_pos = None
        self._peeked_irrelevant_regions = []
        self._reset_frontier_state()
        self._reset_semantic_goal_state()
        if self.hypergraph_reasoner is not None:
            self.hypergraph_reasoner.clear_cache()

    def graph_corr(self, goal, graph):
        prompt = self.prompt_graph_corr_0.format(graph.center_node.caption, goal)
        response_0 = self.llm(prompt=prompt)
        prompt = self.prompt_graph_corr_1.format(graph.center_node.caption, goal)
        response_1 = self.llm(prompt=prompt)
        prompt = self.prompt_graph_corr_2.format(graph.caption, response_1)
        response_2 = self.llm(prompt=prompt)
        prompt = self.prompt_graph_corr_3.format(response_0, response_1 + response_2, graph.center_node.caption, goal)
        response_3 = self.llm(prompt=prompt)
        corr_score = self.text2value(response_3)
        return corr_score
    
    def text2value(self, text):
        try:
            value = float(text)
        except:
            value = 0
        return value
    
    # ========================================================================
    # ==================== 动态时序场景图核心方法 ==============================
    # ========================================================================
    
    def update_fov_masking(self):
        """结算哪些历史节点理应在当前视锥（FOV）内
        基于机器人当前位姿、相机内参（FOV角度）与节点距离粗略判断
        """
        # 机器人格子坐标与朝向
        map_h, map_w = self._full_map_shape()
        agent_col = map_w / 2.0 + (self.observations['gps'][0] * 100.0 / self.map_resolution)
        agent_row_world = map_h / 2.0 - (self.observations['gps'][1] * 100.0 / self.map_resolution)
        agent_row = map_h - 1 - agent_row_world
        agent_theta = (self.observations['compass'] - np.pi / 2)[0]
        
        hfov_rad = np.deg2rad(self.args.hfov)
        # 假设最大感知距离约为 8 米 (转为格子数)
        max_dist_cells = 8.0 * (100.0 / self.map_resolution) 
        
        for node in self.nodes:
            if node.center is None or not self._in_bounds_rc(node.center):
                node.is_in_FOV = False
                continue
            # 如果当前时间步刚被观测到，必然在视野内
            if node.t_last == self.navigate_steps:
                node.is_in_FOV = True
                continue
                
            # ★ 修复：node.center[1] 和 agent_y 均在翻转后的地图坐标系中，
            #   直接在同一坐标系下计算差值，与 agent_theta (compass-π/2) 一致
            d_row = node.center[0] - agent_row
            d_col = node.center[1] - agent_col
            
            dist = np.hypot(d_col, d_row)
            
            # 计算节点相对于机器人的夹角
            angle_to_node = np.arctan2(d_row, d_col)
            angle_diff = (angle_to_node - agent_theta + np.pi) % (2 * np.pi) - np.pi
            
            # 如果位于视锥范围内
            if dist < max_dist_cells and abs(angle_diff) < hfov_rad / 2:
                node.is_in_FOV = True
            else:
                node.is_in_FOV = False

    def anomaly_detector(self):
        """主动异常检测：判断是否存在时序或置信度矛盾"""
        if self.disable_temporal:
            return False, None
        # 新鲜度保护：若当前步 update_scenegraph() 尚未执行，
        # is_in_FOV 是上帧残留值，直接跳过防止误报
        last_obs_step = getattr(self, '_last_obs_step', -1)
        fresh_steps = int(self.temporal_config.get('anomaly_fresh_steps', 2))
        if last_obs_step < 0 or self.navigate_steps - last_obs_step > fresh_steps:
            return False, None
        # 1. 检查置信度矛盾
        for node in self.nodes:
            if (
                node.center is not None
                and self._in_bounds_rc(node.center)
                and node.is_in_FOV
                and node.t_last < self.navigate_steps
            ):
                # ★ 核心修复：只关注高频核心物体，且未被验证过的节点
                if node.c_trend < self.temporal_config['trend_threshold_down'] and node.n_obs > 3:
                    if not getattr(node, 'anomaly_checked', False):
                        node.anomaly_checked = True  # 标记为已验证，防止死锁
                        print(f"        [Anomaly] 节点 {node.caption} 置信度异常下降，触发重验证。")
                        return True, node.center
                    
        # 2. 检查时间悖论 (同理加冷却)
        for edge in self.get_edges():
            if edge.check_temporal_paradox():
                if not getattr(edge, 'anomaly_checked', False):
                    edge.anomaly_checked = True
                    print(f"        [Anomaly] 边 ({edge.node1.caption}, {edge.node2.caption}) 存在时间悖论。")
                    if edge.node1.center is not None and self._in_bounds_rc(edge.node1.center):
                        return True, edge.node1.center
                
        return False, None

    def compute_bayesian_confidence(self, node, current_time):
        """计算节点的贝叶斯时序置信度
        
        公式: P(X_v=1|O_{1:t}) = σ(Σ α^{t-t_k} sim(o_k,v) + β log((t_last-t_first)/t))
        
        Args:
            node: ObjectNode 节点
            current_time: 当前时间步
            alpha: 时间衰减系数 (0,1)，越大衰减越慢
            beta: 时间跨度正则系数
            
        Returns:
            confidence: 节点存在的后验概率 [0,1]
        """
        alpha = self.temporal_config['alpha']
        beta = self.temporal_config['beta']
        gamma = self.temporal_config.get('gamma', 0.05)
        if node.n_obs == 0 or node.t_first is None:
            return 0.5  # 无观测时返回先验
        
        time_weighted_score = 0.0
        weight_sum = 0.0
        for i, obs_time in enumerate(node.observation_times):
            decay = alpha ** (current_time - obs_time)
            sim = node.similarity_scores[i] if i < len(node.similarity_scores) else node.score
            time_weighted_score += decay * sim
            weight_sum += decay
        
        # 归一化：加权平均
        if weight_sum > 0:
            normalized_score = time_weighted_score / weight_sum
        else:
            normalized_score = 0.5
        
        # 存活时间奖励 (仅限稳定存活的节点)
        time_span = node.t_last - node.t_first + 1
        time_bonus = beta * np.log(time_span)
        
        # ★ 核心修复：最近未观测惩罚 (Recency Penalty)
        # 如果一个物体很久没看到了，它的惩罚会越来越大，最终压垮置信度
        time_since_last_obs = current_time - node.t_last
        recency_penalty = gamma * time_since_last_obs
        
        # 中心化 + 缩放 + 奖励 - 惩罚
        logit = (normalized_score - 0.5) * 4 + time_bonus - recency_penalty
        confidence = 1.0 / (1.0 + np.exp(-logit))
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def compute_graph_delta(self, k=10):
        """计算图的时序差分 ΔG_t = G_t ⊖ G_{t-k}
        
        返回:
            delta: dict 包含:
                - new_objects: 新增节点（最近k步内首次观测）
                - vanished_objects: 消失节点（超过遗忘阈值未观测）
                - confirmed_objects: 置信度上升节点
                - dubious_objects: 置信度下降节点
        """
        current_time = self.navigate_steps
        threshold_up = 0.05    # 置信度上升阈值
        threshold_down = -0.05  # 置信度下降阈值
        
        delta = {
            'new_objects': [],       # ΔV_t^+
            'vanished_objects': [],  # ΔV_t^-
            'confirmed_objects': [], # ΔC_t^↑
            'dubious_objects': []    # ΔC_t^↓
        }
        
        for node in self.nodes:
            # 检测新增节点
            if node.t_first is not None and current_time - node.t_first < k:
                delta['new_objects'].append(node.caption)
            
            # 检测消失节点（超过遗忘阈值未观测）
            if node.t_last is not None and current_time - node.t_last > self.T_forget:
                delta['vanished_objects'].append(node.caption)
            
            # 置信度趋势检测
            if node.c_trend > threshold_up:
                delta['confirmed_objects'].append(node.caption)
            elif node.c_trend < threshold_down:
                delta['dubious_objects'].append(node.caption)
        
        self.temporal_graph_delta = delta
        return delta
    
    def compute_temporal_frontier_scores(self, frontier_locations, current_time, 
                                          conf_weight=0.3, novelty_weight=0.2):
        """计算前沿的时序增强评分
        
        公式: P_temporal(F_i) = Σ_j (P_sub_j * conf(v_j) * novelty(F_i)) / D_ij
        
        Args:
            frontier_locations: 前沿位置数组 (N, 2)
            current_time: 当前时间步
            conf_weight: 置信度权重
            novelty_weight: 新颖性权重
            
        Returns:
            temporal_scores: 每个前沿的时序评分数组
        """
        num_frontiers = len(frontier_locations)
        temporal_scores = np.zeros(num_frontiers)
        
        active_nodes = self._active_nodes(require_center=True)
        if len(active_nodes) == 0:
            return temporal_scores
        
        for i, frontier in enumerate(frontier_locations):
            # 1. 基于附近节点置信度的评分
            conf_score = 0.0
            nearby_count = 0
            for node in active_nodes:
                dist = np.linalg.norm(np.array(frontier) - np.array(node.center))
                if dist < 100:  # 100格范围内的节点
                    # 置信度 * 趋势（正趋势加分）
                    conf_contribution = node.confidence * (1 + max(node.c_trend, 0))
                    conf_score += conf_contribution / max(dist, 1)
                    nearby_count += 1
            
            if nearby_count > 0:
                conf_score /= nearby_count
            
            # 2. 新颖性因子（惩罚近期已探索的区域）
            fx, fy = int(frontier[0]), int(frontier[1])
            if 0 <= fx < self.full_width and 0 <= fy < self.full_height:
                last_visit = self.last_visit_map[fx, fy]
                novelty = 1.0 - (last_visit / max(current_time, 1))
            else:
                novelty = 1.0
            
            temporal_scores[i] = conf_weight * conf_score + novelty_weight * novelty
        
        return temporal_scores
    
    def compute_hypergraph_frontier_scores(self, frontier_locations, prob_map, 
                                            prob_weight=0.5):
        """计算前沿的超图概率增强评分
        
        基于 BP 推断输出的后验概率分布，引导机器人前往高概率目标区域。
        
        公式: S_hyper(F_i) = Σ_j (prob_map[v_j] / D(F_i, v_j))
        
        Args:
            frontier_locations: 前沿位置数组 (N, 2)
            prob_map: 节点后验概率分布 {node_id: probability}
            prob_weight: 概率权重
            
        Returns:
            hypergraph_scores: 每个前沿的超图概率评分数组
        """
        num_frontiers = len(frontier_locations)
        hypergraph_scores = np.zeros(num_frontiers)
        
        active_nodes = self._active_nodes(require_center=True)
        if len(active_nodes) == 0 or not prob_map:
            return hypergraph_scores
            
        # ★ 修复：使用 uid 获取准确的后验概率
        node_probs = {}
        for node in active_nodes:
            uid = getattr(self, 'node_uid_map', {}).get(node, None)
            if uid:
                node_probs[node] = prob_map.get(uid, 0.0)
            
        for i, frontier in enumerate(frontier_locations):
            prob_score = 0.0
            
            for node, prob in node_probs.items():
                if node.center is not None and self._in_bounds_rc(node.center) and prob > 0.1:
                    dist = np.linalg.norm(np.array(frontier) - np.array(node.center))
                    # 距离越近、概率越高，得分越高
                    if dist < 150:  # 150格范围内
                        prob_score += prob / max(dist, 1)
                        
            hypergraph_scores[i] = prob_weight * prob_score
            
        # 归一化到合理范围
        if np.max(hypergraph_scores) > 0:
            hypergraph_scores = hypergraph_scores / np.max(hypergraph_scores) * 2.0
            
        return hypergraph_scores
    
    def get_nearby_nodes(self, position, radius=50):
        """获取指定位置附近的节点
        
        Args:
            position: (row, col) 位置
            radius: 搜索半径
            
        Returns:
            nearby_nodes: 附近的节点列表
        """
        nearby_nodes = []
        for node in self._active_nodes(require_center=True):
            dist = np.linalg.norm(np.array(position) - np.array(node.center))
            if dist < radius:
                nearby_nodes.append(node)
        return nearby_nodes
    
    def update_visit_map(self, position, current_time):
        """更新位置访问时间记录
        
        Args:
            position: (row, col) 当前位置
            current_time: 当前时间步
        """
        row, col = int(position[0]), int(position[1])
        map_h, map_w = self._full_map_shape()
        # 更新当前位置及周围区域的访问时间
        for d_row in range(-5, 6):
            for d_col in range(-5, 6):
                n_row, n_col = row + d_row, col + d_col
                if 0 <= n_row < map_h and 0 <= n_col < map_w:
                    self.last_visit_map[n_row, n_col] = current_time
    
    def detect_inconsistencies(self):
        """检测场景图中的不一致性
        
        1. 时间悖论：边的创建时间早于节点
        2. 置信度不一致：节点置信度下降但邻居上升
        
        Returns:
            inconsistencies: dict 包含检测到的各类不一致
        """
        inconsistencies = {
            'temporal_paradox_edges': [],
            'confidence_inconsistent_nodes': []
        }
        
        # 检测边的时间悖论
        for edge in self.get_edges():
            if edge.check_temporal_paradox():
                inconsistencies['temporal_paradox_edges'].append({
                    'edge': (edge.node1.caption, edge.node2.caption),
                    'edge_create_time': edge.t_create,
                    'node1_first_obs': edge.node1.t_first,
                    'node2_first_obs': edge.node2.t_first
                })
        
        # 检测置信度不一致
        threshold = 0.05
        for node in self.nodes:
            if node.c_trend < -threshold:
                # 检查邻居节点趋势
                neighbor_trends_up = False
                for edge in node.edges:
                    neighbor = edge.node1 if edge.node2 == node else edge.node2
                    if neighbor.c_trend > threshold:
                        neighbor_trends_up = True
                        break
                
                if neighbor_trends_up:
                    inconsistencies['confidence_inconsistent_nodes'].append({
                        'node': node.caption,
                        'trend': node.c_trend,
                        'suggestion': 'Consider re-detecting this subgraph'
                    })
        
        return inconsistencies
    
    def get_temporal_aware_prompt(self, goal):
        """生成时序感知的 LLM Prompt
        
        Args:
            goal: 目标描述
            
        Returns:
            prompt: 包含时序变化信息的prompt
        """
        # 计算图差分
        delta = self.compute_graph_delta(k=10)
        
        prompt = f'''Given the scene graph and temporal changes:
Scene Graph: {self.scenegraph}

Temporal Delta (recent changes):
- New objects (just discovered): {delta['new_objects']}
- Vanished objects (not seen recently): {delta['vanished_objects']}
- Confirmed objects (confidence rising): {delta['confirmed_objects']}
- Dubious objects (confidence falling): {delta['dubious_objects']}

Target: {goal}

Based on the temporal information, which area should be explored next to find the target?
Consider: newly discovered objects may indicate unexplored areas; confirmed objects are reliable landmarks.
'''
        return prompt
    
    def filter_low_confidence_nodes(self, threshold=0.3):
        """过滤低置信度节点（用于减少LLM prompt长度）
        
        Args:
            threshold: 置信度阈值
            
        Returns:
            high_conf_nodes: 高置信度节点列表
        """
        return self._active_nodes(threshold=threshold)
    
    def get_effective_graph_size(self):
        """获取有效图规模（过滤遗忘节点后）
        
        用于评估时序遗忘机制的效果
        """
        return len(self._active_nodes(threshold=0.0))
