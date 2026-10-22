"""
超图推理与信念传播模块 (Hypergraph Belief Propagation)

核心功能：
1. HyperEdge: 超边数据结构，表示高阶语义关系（功能区）
2. FactorGraph: 因子图构建，包含一元/二元/高阶因子
3. BeliefPropagation: BP 推断引擎，迭代消息传递
4. HyperGraphReasoner: 主推理器，集成目标解析与超图生成

设计原则：
- 替代原有 overlap.py 的硬匹配（GED），改用软概率推断
- 与动态时序场景图无缝集成，利用 node.confidence 作为因子权重
- 支持 Object/Image/Text 三种输入模态的统一解析
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import re


@dataclass
class HyperEdge:
    """超边：连接多个节点的高阶语义关系
    
    例如：sleeping_area 超边连接 [bed, nightstand, lamp]
    """
    name: str                          # 超边名称（功能区名）
    node_ids: Set[str]                 # 包含的节点ID集合
    confidence: float = 1.0            # 超边置信度
    source: str = "llm"                # 来源：llm / rule / observation
    
    def contains(self, node_id: str) -> bool:
        return node_id in self.node_ids
    
    def overlap_ratio(self, other_nodes: Set[str]) -> float:
        """计算与给定节点集的重叠比例"""
        if len(self.node_ids) == 0:
            return 0.0
        intersection = self.node_ids & other_nodes
        return len(intersection) / len(self.node_ids)


@dataclass 
class UnaryFactor:
    """一元因子：节点的个体视觉/语义特征匹配度
    
    ψ_v(x_v) = CLIP_sim(scene_node, goal_template)
    """
    node_id: str
    potential: float  # [0, 1] 语义相似度
    
    
@dataclass
class BinaryFactor:
    """二元因子：两节点间的空间拓扑约束
    
    ψ_{vv'}(x_v, x_v') = spatial_consistency(edge_scene, edge_goal)
    """
    node_id_1: str
    node_id_2: str
    relation_type: str  # "next_to", "on", "behind" 等
    potential: float    # [0, 1] 关系一致性得分


@dataclass
class HyperFactor:
    """高阶超边因子：全局功能区模式一致性
    
    ψ_e(x_e) = pattern_match(local_subgraph, functional_zone)
    """
    hyperedge: HyperEdge
    matched_nodes: Set[str]  # 场景图中匹配到的节点
    potential: float         # [0, 1] 模式一致性得分


class FactorGraph:
    """因子图：组织所有因子用于 BP 推断
    
    结构：
    - 变量节点：场景图中的每个 ObjectNode
    - 因子节点：一元、二元、高阶因子
    """
    
    def __init__(self):
        self.variable_nodes: Dict[str, float] = {}  # node_id -> prior belief
        self.unary_factors: List[UnaryFactor] = []
        self.binary_factors: List[BinaryFactor] = []
        self.hyper_factors: List[HyperFactor] = []
        
        # 消息缓存（用于 BP 迭代）
        self.messages_to_var: Dict[str, List[float]] = {}   # node_id -> [msg1, msg2, ...]
        self.messages_from_var: Dict[str, List[float]] = {}
        
    def add_variable(self, node_id: str, prior: float = 0.5):
        """添加变量节点"""
        self.variable_nodes[node_id] = prior
        self.messages_to_var[node_id] = []
        self.messages_from_var[node_id] = []
        
    def add_unary_factor(self, node_id: str, potential: float):
        """添加一元因子"""
        if node_id in self.variable_nodes:
            self.unary_factors.append(UnaryFactor(node_id, potential))
            
    def add_binary_factor(self, node_id_1: str, node_id_2: str, 
                          relation_type: str, potential: float):
        """添加二元因子"""
        if node_id_1 in self.variable_nodes and node_id_2 in self.variable_nodes:
            self.binary_factors.append(BinaryFactor(
                node_id_1, node_id_2, relation_type, potential
            ))
            
    def add_hyper_factor(self, hyperedge: HyperEdge, 
                         matched_nodes: Set[str], potential: float):
        """添加高阶超边因子"""
        self.hyper_factors.append(HyperFactor(hyperedge, matched_nodes, potential))
        
    def get_neighbors(self, node_id: str) -> Set[str]:
        """获取节点的邻居（通过二元因子连接的节点）"""
        neighbors = set()
        for bf in self.binary_factors:
            if bf.node_id_1 == node_id:
                neighbors.add(bf.node_id_2)
            elif bf.node_id_2 == node_id:
                neighbors.add(bf.node_id_1)
        return neighbors


class BeliefPropagation:
    """信念传播推断引擎
    
    在因子图上执行 Sum-Product 消息传递算法，
    计算每个变量节点的边缘后验概率。
    """
    
    def __init__(self, factor_graph: FactorGraph, 
                 max_iterations: int = 15,
                 convergence_threshold: float = 1e-4,
                 damping: float = 0.5):
        """
        Args:
            factor_graph: 因子图
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
            damping: 阻尼系数（防止振荡）
        """
        self.fg = factor_graph
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.damping = damping
        
        # 信念状态
        self.beliefs: Dict[str, float] = {}
        self._init_beliefs()
        
    def _init_beliefs(self):
        """初始化信念为先验"""
        for node_id, prior in self.fg.variable_nodes.items():
            self.beliefs[node_id] = prior
            
    def _compute_unary_message(self, factor: UnaryFactor) -> float:
        """计算一元因子传递给变量的消息"""
        return float(np.clip(factor.potential, 0.01, 0.99))
    
    def _compute_binary_message(self, factor: BinaryFactor, 
                                 to_node: str) -> float:
        """计算二元因子传递给变量的消息
        
        m_{f→v}(x_v) = Σ_{x_v'} ψ(x_v, x_v') * b(x_v')
        """
        from_node = factor.node_id_1 if factor.node_id_2 == to_node else factor.node_id_2
        from_belief = self.beliefs.get(from_node, 0.5)
        
        # 简化：二值变量的边缘化
        # 如果 from_node 存在(belief高)且关系一致，则传递正面消息
        potential = float(np.clip(factor.potential, 0.01, 0.99))
        message = potential * from_belief + (1 - potential) * (1 - from_belief)
        return float(np.clip(message, 0.01, 0.99))
    
    def _compute_hyper_message(self, factor: HyperFactor, 
                                to_node: str) -> float:
        """计算高阶超边因子传递给变量的消息
        
        关键：如果超边中的其他节点已被高置信度匹配，
        则即使当前节点漏检，也传递较强的正面消息（局部脑补）。
        """
        if to_node not in factor.matched_nodes:
            return 0.5  # 不在匹配集中，无消息
            
        # 计算超边中其他节点的平均置信度
        other_nodes = factor.matched_nodes - {to_node}
        if len(other_nodes) == 0:
            return float(np.clip(factor.potential, 0.01, 0.99))
            
        other_beliefs = [self.beliefs.get(n, 0.5) for n in other_nodes]
        avg_other_belief = np.mean(other_beliefs)
        
        # 超边模式一致性 + 其他节点支持 → 增强当前节点信念
        # 这是 "局部脑补" 的核心：即使漏检，邻居支持也能拉高概率
        potential = float(np.clip(factor.potential, 0.01, 0.99))
        message = potential * (0.3 + 0.7 * avg_other_belief)
        return float(np.clip(message, 0.01, 0.99))
    
    def _update_belief(self, node_id: str) -> float:
        """更新单个节点的信念
        
        b(x_v) ∝ ψ_v(x_v) * Π_{f∈N(v)} m_{f→v}(x_v)
        """
        # 收集所有传入消息
        messages = []
        
        # 一元因子消息
        for uf in self.fg.unary_factors:
            if uf.node_id == node_id:
                messages.append(self._compute_unary_message(uf))
                
        # 二元因子消息
        for bf in self.fg.binary_factors:
            if bf.node_id_1 == node_id or bf.node_id_2 == node_id:
                messages.append(self._compute_binary_message(bf, node_id))
                
        # 高阶超边因子消息
        for hf in self.fg.hyper_factors:
            if node_id in hf.matched_nodes:
                messages.append(self._compute_hyper_message(hf, node_id))
        
        if len(messages) == 0:
            return self.fg.variable_nodes.get(node_id, 0.5)
            
        # 乘积归一化
        eps = 1e-4
        prior = float(np.clip(self.fg.variable_nodes.get(node_id, 0.5), eps, 1.0 - eps))
        logit = np.log(prior / (1.0 - prior))
        for message in messages:
            m = float(np.clip(message, eps, 1.0 - eps))
            logit += np.log(m / (1.0 - m))
        log_belief = np.clip(logit, -30.0, 30.0)
        new_belief = 1.0 / (1.0 + np.exp(-log_belief))  # sigmoid 归一化
        
        # 阻尼平滑
        old_belief = self.beliefs.get(node_id, 0.5)
        damped_belief = self.damping * old_belief + (1 - self.damping) * new_belief
        
        return float(np.clip(damped_belief, 0.01, 0.99))
    
    def run(self) -> Dict[str, float]:
        """执行 BP 推断
        
        Returns:
            prob_map: {node_id: posterior_probability}
        """
        for iteration in range(self.max_iterations):
            max_delta = 0.0
            
            # 更新所有变量节点的信念
            new_beliefs = {}
            for node_id in self.fg.variable_nodes:
                new_belief = self._update_belief(node_id)
                delta = abs(new_belief - self.beliefs.get(node_id, 0.5))
                max_delta = max(max_delta, delta)
                new_beliefs[node_id] = new_belief
                
            self.beliefs = new_beliefs
            
            # 收敛检测
            if max_delta < self.convergence_threshold:
                print(f"        [BP] Converged at iteration {iteration + 1}")
                break
                
        return self.beliefs


class HyperGraphReasoner:
    """超图推理器：整合目标解析、超边生成与 BP 推断
    
    替代原有 overlap.py 中的 GraphMatcher 硬匹配逻辑。
    """
    
    # LLM Prompt 模板
    PROMPT_GOAL_EXPAND = """Given the target object "{goal}", list 3-5 objects that typically co-occur with it in a home environment.
Prefer discriminative, navigationally useful objects. Do NOT include generic structural/background objects such as wall, floor, ceiling, window, curtain, picture, painting, door, carpet, or room.
Output format: ["object1", "object2", "object3"]
Only output the JSON array, no explanation."""

    PROMPT_HYPEREDGE_GEN = """Given the target object "{goal}" and these candidate objects: {candidates}
Identify which objects form a coherent functional zone (e.g., sleeping_area, kitchen_counter, living_room_seating).
Output format: {{"zone_name": "zone_type", "members": ["obj1", "obj2", ...]}}
If no coherent zone exists, output: {{"zone_name": null, "members": []}}
Only output JSON, no explanation."""

    PROMPT_IMAGE_PARSE = """Describe the main object in this image and list other objects that appear nearby.
Output format: {{"main_object": "object_name", "nearby_objects": ["obj1", "obj2"], "spatial_relations": ["obj1 is on table", "obj2 is next to chair"]}}"""

    PROMPT_TEXT_PARSE = """Extract the target object and spatial constraints from this navigation instruction: "{text}"
Output format: {{"target": "object_name", "constraints": ["constraint1", "constraint2"], "context_objects": ["obj1", "obj2"]}}"""

    STOP_WORDS = {
        "a", "an", "the", "and", "or", "with", "near", "nearby", "next", "to",
        "small", "large", "big", "little", "wooden", "metal", "plastic",
        "white", "black", "blue", "red", "green", "yellow", "brown", "gray", "grey",
    }
    ALIASES = {
        "couch": "sofa",
        "settee": "sofa",
        "television": "tv",
        "monitor": "tv",
        "tvmonitor": "tv",
        "tv_monitor": "tv",
        "fridge": "refrigerator",
        "freezer": "refrigerator",
        "nightstand": "bedside_table",
        "bedside": "bedside_table",
        "bookshelf": "shelf",
        "bookcase": "shelf",
        "cabinetry": "cabinet",
        "cupboard": "cabinet",
        "drawer": "cabinet",
        "flowerpot": "plant",
        "vase": "vase",
        "microwave": "microwave",
        "seat": "chair",
        "armchair": "chair",
        "stool": "chair",
    }

    LOW_VALUE_CONTEXT_OBJECTS = {
        "wall", "floor", "ceiling", "window", "windows", "curtain", "curtains",
        "picture", "painting", "painting picture", "door", "carpet", "rug",
        "room", "corner", "hall", "area", "light", "object", "item",
    }

    TARGET_CONTEXT_PRIORS = {
        "chair": ["table", "desk", "dining table", "sofa", "counter", "tv"],
        "sofa": ["chair", "coffee table", "tv", "lamp"],
        "bed": ["nightstand", "bedside table", "pillow", "lamp", "dresser"],
        "toilet": ["sink", "shower", "bathtub", "towel", "cabinet"],
        "sink": ["toilet", "mirror", "counter", "cabinet", "towel"],
        "tv": ["sofa", "chair", "table", "cabinet"],
        "plant": ["table", "sofa", "window", "shelf"],
    }

    def __init__(self, llm_func, vlm_func=None, clip_model=None, config: dict = None):
        """
        Args:
            llm_func: LLM 调用函数 (prompt) -> response
            vlm_func: VLM 调用函数 (prompt, image) -> response（可选）
            clip_model: CLIP 模型（用于语义相似度计算）
            config: 配置参数
        """
        self.llm = llm_func
        self.vlm = vlm_func
        self.clip_model = clip_model
        
        # 默认配置
        self.config = {
            'bp_max_iterations': 15,
            'bp_damping': 0.5,
            'clip_sim_threshold': 0.6,
            'hyperedge_min_overlap': 0.34,
            'unary_weight': 1.0,
            'binary_weight': 0.8,
            'hyper_weight': 1.2,
            'prob_threshold': 0.12,
            'prob_keep_topk': 3,
            'pending_edge_scale': 0.35,
            'background_unary_scale': 0.25,
            'background_unary_cap': 0.18,
        }
        if config:
            self.config.update(config)
            
        # 缓存（避免重复 LLM 调用）
        self._hyperedge_cache: Dict[str, List[HyperEdge]] = {}
        self._goal_expand_cache: Dict[str, List[str]] = {}
        
    def parse_goal(self, goal_input: Any, goal_type: str = "object") -> dict:
        """统一目标解析：将 Object/Image/Text 转换为结构化目标模板
        
        Args:
            goal_input: 输入（字符串/图像/文本）
            goal_type: "object" | "image" | "text"
            
        Returns:
            goal_template: {
                "target": str,           # 主目标
                "context": [str],        # 上下文物体
                "constraints": [str],    # 空间约束
            }
        """
        if goal_type == "object":
            # 简单物体名 → LLM 扩展上下文
            target = str(goal_input).lower().strip()
            context = self._expand_goal_context(target)
            return {
                "target": target,
                "context": context,
                "constraints": []
            }
            
        elif goal_type == "image":
            # 图像 → VLM 解析
            if self.vlm is None:
                return {"target": "unknown", "context": [], "constraints": []}
            try:
                response = self.vlm(self.PROMPT_IMAGE_PARSE, goal_input)
                parsed = json.loads(response)
                target = parsed.get("main_object", "unknown")
                return {
                    "target": target,
                    "context": self._filter_context_candidates(
                        parsed.get("nearby_objects", []), target=target, limit=5
                    ),
                    "constraints": parsed.get("spatial_relations", [])
                }
            except Exception as e:
                print(f"        [HyperGraph] WARNING: VLM image parse failed: {e}")
                return {"target": "unknown", "context": [], "constraints": []}
                
        elif goal_type == "text":
            # 文本描述 → LLM 解析
            try:
                prompt = self.PROMPT_TEXT_PARSE.format(text=goal_input)
                response = self.llm(prompt)
                parsed = json.loads(response)
                target = parsed.get("target", "unknown")
                return {
                    "target": target,
                    "context": self._filter_context_candidates(
                        parsed.get("context_objects", []), target=target, limit=5
                    ),
                    "constraints": parsed.get("constraints", [])
                }
            except Exception as e:
                print(f"        [HyperGraph] WARNING: LLM text parse failed: {e}")
                return {"target": str(goal_input), "context": [], "constraints": []}
                
        return {"target": str(goal_input), "context": [], "constraints": []}
    
    def _expand_goal_context(self, goal: str) -> List[str]:
        """使用 LLM 扩展目标物体的上下文共现物体"""
        goal = self._canonical_text(goal) or str(goal).lower().strip()
        if goal in self._goal_expand_cache:
            return self._goal_expand_cache[goal]

        context = []
        context.extend(self.TARGET_CONTEXT_PRIORS.get(goal, []))
            
        try:
            prompt = self.PROMPT_GOAL_EXPAND.format(goal=goal)
            response = self.llm(prompt)
            # 解析 JSON 数组
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                context.extend(json.loads(match.group()))
        except Exception:
            pass
        context = self._filter_context_candidates(context, target=goal, limit=5)
        self._goal_expand_cache[goal] = context
        return context

    def _is_low_value_context(self, name: str) -> bool:
        canonical = self._canonical_text(name)
        if not canonical:
            return True
        tokens = set(canonical.split())
        low_value = {
            self._canonical_text(item)
            for item in self.LOW_VALUE_CONTEXT_OBJECTS
            if self._canonical_text(item)
        }
        return canonical in low_value or bool(tokens & low_value)

    def _filter_context_candidates(self, names: List[str], target: str = None,
                                   limit: int = None) -> List[str]:
        target_c = self._canonical_text(target) if target else ""
        seen = set()
        filtered = []
        for name in names:
            canonical = self._canonical_text(name)
            if not canonical or canonical == target_c or canonical in seen:
                continue
            if self._is_low_value_context(canonical):
                continue
            seen.add(canonical)
            filtered.append(canonical)
            if limit is not None and len(filtered) >= limit:
                break
        return filtered

    def _canonical_text(self, text: str) -> str:
        text = str(text or "").lower().replace("_", " ").replace("-", " ")
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        tokens = []
        for token in text.split():
            stem = self._stem(token)
            stem = self.ALIASES.get(stem, stem)
            if stem and stem not in self.STOP_WORDS:
                tokens.append(stem)
        return " ".join(tokens)

    def _caption_tokens(self, text: str) -> Set[str]:
        return set(self._canonical_text(text).split())

    def _semantic_match_score(self, left: str, right: str) -> float:
        left_tokens = self._caption_tokens(left)
        right_tokens = self._caption_tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0

        if left_tokens == right_tokens:
            return 1.0
        if left_tokens <= right_tokens or right_tokens <= left_tokens:
            return 0.9
        overlap = left_tokens & right_tokens
        if not overlap:
            return 0.0
        short_ratio = len(overlap) / max(1, min(len(left_tokens), len(right_tokens)))
        jaccard = len(overlap) / max(1, len(left_tokens | right_tokens))
        return float(max(0.55 * short_ratio, 0.75 * jaccard))

    def _dedupe_by_canonical(self, names: List[str], limit: int = None) -> List[str]:
        seen = set()
        deduped = []
        for name in names:
            canonical = self._canonical_text(name)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            deduped.append(canonical)
            if limit is not None and len(deduped) >= limit:
                break
        return deduped
    
    def generate_hyperedges(self, goal_template: dict, 
                            scene_nodes: List[dict]) -> List[HyperEdge]:
        """动态生成超边：基于 LLM 推理功能区
        
        Args:
            goal_template: 目标模板
            scene_nodes: 场景图节点列表 [{"id": str, "caption": str, ...}, ...]
            
        Returns:
            hyperedges: 超边列表
        """
        target = self._canonical_text(goal_template.get("target", ""))
        context = self._filter_context_candidates(goal_template.get("context", []), target, limit=6)
        raw_scene_captions = self._dedupe_by_canonical(
            [n.get("caption", "") for n in scene_nodes if n.get("caption", "")],
            limit=12,
        )
        scene_captions = [
            cap for cap in raw_scene_captions
            if cap == target or not self._is_low_value_context(cap)
        ]
        cache_key = json.dumps(
            {"target": target, "context": context, "scene": scene_captions},
            sort_keys=True,
        )
        
        if cache_key in self._hyperedge_cache:
            return self._hyperedge_cache[cache_key]
            
        hyperedges = []
        
        # 收集场景中的物体名称
        
        # 候选物体：目标 + 上下文 + 场景中存在的物体
        candidates = self._dedupe_by_canonical([target] + context + scene_captions, limit=12)
        
        if len(candidates) < 2:
            self._hyperedge_cache[cache_key] = []
            return []
            
        # 调用 LLM 生成功能区超边
        try:
            prompt = self.PROMPT_HYPEREDGE_GEN.format(
                goal=target, 
                candidates=candidates
            )
            response = self.llm(prompt)
            
            # 解析 JSON
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                result = json.loads(match.group())
                zone_name = result.get("zone_name")
                members = result.get("members", [])
                
                member_ids = {
                    self._canonical_text(m)
                    for m in members
                    if self._canonical_text(m)
                }
                member_ids = {
                    m for m in member_ids
                    if m == target or not self._is_low_value_context(m)
                }
                if target and target not in member_ids:
                    print(
                        f"        [HyperGraph] Hyperedge '{zone_name}' rejected: "
                        f"members do not include target '{target}'"
                    )
                    member_ids = set()
                if zone_name and len(member_ids) >= 2:
                    hyperedge = HyperEdge(
                        name=zone_name,
                        node_ids=member_ids,
                        confidence=0.9,
                        source="llm"
                    )
                    hyperedges.append(hyperedge)
        except Exception as e:
            print(f"        [HyperGraph] Failed to generate hyperedge: {e}")

        fallback_members = self._dedupe_by_canonical([target] + context, limit=6)
        if not hyperedges and len(fallback_members) >= 2:
            hyperedges.append(HyperEdge(
                name=f"{target or 'goal'}_context",
                node_ids=set(fallback_members),
                confidence=0.55,
                source="rule"
            ))
            
        self._hyperedge_cache[cache_key] = hyperedges
        return hyperedges
    
    @staticmethod
    def _stem(word: str) -> str:
        """简易词干提取：去除常见英文后缀以匹配单复数/变体"""
        w = word.lower().strip()
        if w in {'dining', 'living', 'ceiling', 'painting'}:
            return w
        for suffix in ('ies', 'es', 'ing', 's'):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                stem = w[:-len(suffix)]
                if suffix == 'ies':
                    stem += 'y'  # batteries → battery
                return stem
        return w
    
    def compute_clip_similarity(self, scene_node: dict, 
                                 goal_target: str) -> float:
        """计算场景节点与目标的语义相似度
        
        如果没有 CLIP 模型，则使用增强的字符串匹配（支持词干/复数匹配）。
        """
        caption = scene_node.get("caption", "").lower().strip()
        target = goal_target.lower().strip()
        
        if not caption or not target:
            return 0.0

        caption_tokens = self._caption_tokens(caption)
        target_tokens = self._caption_tokens(target)
        if caption_tokens and target_tokens:
            if caption_tokens == target_tokens:
                return 1.0
            if target_tokens <= caption_tokens or caption_tokens <= target_tokens:
                return 0.9
            overlap = caption_tokens & target_tokens
            if overlap:
                short_ratio = len(overlap) / max(1, min(len(caption_tokens), len(target_tokens)))
                jaccard = len(overlap) / max(1, len(caption_tokens | target_tokens))
                return float(max(0.6 + 0.2 * short_ratio, 0.75 * jaccard))
        
        # 1. 精确匹配
        if caption == target:
            return 1.0
        
        # 2. 词干匹配（处理 windows↔window, chairs↔chair 等）
        caption_stem = self._stem(caption)
        target_stem = self._stem(target)
        if caption_stem == target_stem:
            return 0.95
            
        # 3. 子串匹配（含词干）
        if (target in caption or caption in target or
            target_stem in caption or caption_stem in target or
            target_stem in caption_stem or caption_stem in target_stem):
            return 0.8
        
        # 4. 词级别匹配（处理多词短语，如 "window with blinds" 包含 "window"）
        words_caption = set(caption.split())
        words_target = set(target.split())
        stems_caption = {self._stem(w) for w in words_caption}
        stems_target = {self._stem(w) for w in words_target}
        
        # 词干交集
        stem_overlap = stems_caption & stems_target
        if stem_overlap:
            # 交集占较短集合的比例
            ratio = len(stem_overlap) / min(len(stems_caption), len(stems_target))
            return 0.6 + 0.2 * ratio  # 0.6 ~ 0.8
        
        # 5. 单词子串匹配（caption 的某个词是 target 某个词的子串，反之亦然）
        for wc in stems_caption:
            for wt in stems_target:
                if len(wc) >= 3 and len(wt) >= 3:
                    if wc in wt or wt in wc:
                        return 0.5
        
        # 6. Jaccard 回退
        if len(stems_caption | stems_target) == 0:
            return 0.0
        jaccard = len(stems_caption & stems_target) / len(stems_caption | stems_target)
        return jaccard * 0.6
    
    def build_factor_graph(self, scene_nodes: List[dict],
                           scene_edges: List[dict],
                           goal_template: dict,
                           hyperedges: List[HyperEdge],
                           node_confidences: Dict[str, float] = None) -> FactorGraph:
        """构建因子图
        
        Args:
            scene_nodes: 场景图节点
            scene_edges: 场景图边
            goal_template: 目标模板
            hyperedges: 超边列表
            node_confidences: 节点时序置信度（来自动态时序场景图）
            
        Returns:
            factor_graph: 构建好的因子图
        """
        fg = FactorGraph()
        target = goal_template["target"]
        context = set(goal_template.get("context", []))
        
        # 1. 添加变量节点（场景图中的每个节点）
        # ★ 修复：prior 应为无信息先验 0.5（"该节点是否为目标"），
        #   temporal confidence 表示"该节点是否存在"，是正交问题，
        #   应作为一元因子权重的调制系数，而非 BP 先验。
        for node in scene_nodes:
            node_id = node.get("id", node.get("caption", ""))
            fg.add_variable(node_id, 0.5)
            
        # 2. 添加一元因子（语义相似度）
        target_canonical = self._canonical_text(target)
        for node in scene_nodes:
            node_id = node.get("id", node.get("caption", ""))
            caption_canonical = self._canonical_text(node.get("caption", ""))
            
            # 与主目标的相似度
            sim_target = self.compute_clip_similarity(node, target)
            
            # 与上下文的最大相似度
            sim_context = 0.0
            for ctx in context:
                sim_ctx = self.compute_clip_similarity(node, ctx)
                sim_context = max(sim_context, sim_ctx)
                
            # 综合一元势能
            potential = max(sim_target, sim_context * 0.7)
            # ★ 修复：用 temporal confidence 调制一元势能（存在置信度高 → 匹配更可信）
            temporal_conf = node_confidences.get(node_id, 0.5) if node_confidences else 0.5
            potential = potential * (0.5 + 0.5 * temporal_conf)  # 0.5~1.0 调制
            potential = potential * self.config['unary_weight']
            if (
                self._is_low_value_context(caption_canonical)
                and caption_canonical != target_canonical
                and target_canonical not in set(caption_canonical.split())
            ):
                potential = min(
                    potential * float(self.config.get('background_unary_scale', 0.25)),
                    float(self.config.get('background_unary_cap', 0.18)),
                )
            
            if potential > 0.1:
                fg.add_unary_factor(node_id, potential)
            else:
                # ★ 修复：无语义匹配的节点添加低势能一元因子（负面证据），
                #   防止所有节点因缺少排斥力而一致趋向高概率
                fg.add_unary_factor(node_id, 0.1)
                
        # 3. 添加二元因子（空间拓扑）
        goal_constraints = goal_template.get("constraints", [])
        for edge in scene_edges:
            source = edge.get("source", "")
            target_node = edge.get("target", "")
            relation = edge.get("type", "")
            source_caption = edge.get("source_caption", source)
            target_caption = edge.get("target_caption", target_node)
            status = edge.get("status", "active")
            if status == "rejected" or not relation:
                continue
            
            # 检查是否满足目标约束
            constraint_match = 0.5  # 默认中性
            for constraint in goal_constraints:
                constraint_l = constraint.lower()
                if (
                    source_caption.lower() in constraint_l
                    or target_caption.lower() in constraint_l
                    or relation.lower() in constraint_l
                ):
                    constraint_match = 0.8
                    break

            source_low_value = self._is_low_value_context(source_caption)
            target_low_value = self._is_low_value_context(target_caption)
            if constraint_match <= 0.5 and source_low_value and target_low_value:
                continue
                    
            edge_weight = float(edge.get("weight", 0.5))
            edge_confidence = float(edge.get("confidence", edge_weight))
            geometry_score = float(edge.get("geometry_score", edge_confidence))
            edge_support = np.clip(
                0.45 * edge_confidence + 0.35 * edge_weight + 0.20 * geometry_score,
                0.0,
                1.0
            )
            if status == "pending":
                edge_support *= float(self.config.get('pending_edge_scale', 0.35))
            potential = constraint_match * edge_support * self.config['binary_weight']
            if constraint_match <= 0.5 and (source_low_value or target_low_value):
                potential *= float(self.config.get('background_unary_scale', 0.25))
            if potential <= 0.05:
                continue
            fg.add_binary_factor(source, target_node, relation, potential)
            
        # 4. 添加高阶超边因子
        # ★ 修复：建立 caption → UID 映射，确保 matched_nodes 使用 UID
        #   hyperedge.node_ids 包含 caption 级别字符串（如 "bed"），
        #   但因子图变量节点使用 UID（如 "bed_140234567890"），
        #   必须将匹配结果从 caption 转换为 UID，否则超边因子完全失效。
        caption_entries = []
        for n in scene_nodes:
            cap = n.get("caption", "").lower()
            uid = n.get("id", cap)
            canonical = self._canonical_text(cap)
            if canonical:
                caption_entries.append((cap, canonical, uid))

        match_threshold = min(float(self.config.get('clip_sim_threshold', 0.6)), 0.6)
        for hyperedge in hyperedges:
            matched_members = set()
            matched_uids = set()
            for member in hyperedge.node_ids:
                best_score = 0.0
                best_uids = []
                for _cap, canonical, uid in caption_entries:
                    score = self._semantic_match_score(member, canonical)
                    if score > best_score + 1e-6:
                        best_score = score
                        best_uids = [uid]
                    elif abs(score - best_score) <= 1e-6 and score > 0:
                        best_uids.append(uid)
                if best_score >= match_threshold:
                    matched_members.add(member)
                    matched_uids.update(best_uids)

            overlap = len(matched_members) / max(len(hyperedge.node_ids), 1)
            if len(matched_members) >= 2 and overlap >= float(self.config['hyperedge_min_overlap']):
                potential = hyperedge.confidence * overlap * self.config['hyper_weight']
                potential = float(np.clip(potential, 0.01, 0.99))
                fg.add_hyper_factor(hyperedge, matched_uids, potential)
            elif hyperedge.node_ids:
                print(
                    f"        [HyperGraph] Hyperedge '{hyperedge.name}' skipped: "
                    f"matched={sorted(matched_members)}, members={sorted(hyperedge.node_ids)}, "
                    f"overlap={overlap:.2f}"
                )

        hyperedges = []
        caption_to_uids = {}
        scene_captions = set()
        for hyperedge in hyperedges:
            matched_captions = hyperedge.node_ids & scene_captions
            if len(matched_captions) >= 2:
                overlap = hyperedge.overlap_ratio(scene_captions)
                if overlap >= self.config['hyperedge_min_overlap']:
                    # 将匹配的 caption 展开为对应的 UID 集合
                    matched_uids = set()
                    for cap in matched_captions:
                        matched_uids.update(caption_to_uids[cap])
                    potential = hyperedge.confidence * overlap * self.config['hyper_weight']
                    fg.add_hyper_factor(hyperedge, matched_uids, potential)
                    
        return fg
    
    def infer(self, scene_graph: dict, 
              goal_input: Any,
              goal_type: str = "object",
              node_confidences: Dict[str, float] = None,
              goal_template: dict = None) -> tuple:
        """执行超图推断
        
        这是主入口函数，替代原有 overlap() 硬匹配。
        
        Args:
            scene_graph: 场景图 {"nodes": [...], "edges": [...]}
            goal_input: 目标输入
            goal_type: "object" | "image" | "text"
            node_confidences: 节点时序置信度（可选，来自动态时序场景图）
            goal_template: 预构建的目标模板（可选）。若提供则跳过 parse_goal，
                          避免对已解析的目标重复调用 VLM/LLM。
            
        Returns:
            (prob_map, goal_template): prob_map = {node_id: posterior_probability},
                                       goal_template = 本次使用的目标模板
            
        使用示例:
            reasoner = HyperGraphReasoner(llm_func, vlm_func)
            prob_map, tmpl = reasoner.infer(scene_graph, "bed", "object", node_confidences)
            # prob_map = {"bed": 0.92, "nightstand": 0.78, "lamp": 0.65, ...}
        """
        print("        [HyperGraph] Starting probabilistic inference...")
        
        # Step 1: 目标统一解析（若调用方已提供 goal_template 则直接复用）
        if goal_template is None:
            goal_template = self.parse_goal(goal_input, goal_type)
        print(f"        [HyperGraph] Goal template: {goal_template['target']} + {goal_template['context'][:3]}")
        
        # Step 2: 动态超边生成
        scene_nodes = scene_graph.get("nodes", [])
        hyperedges = self.generate_hyperedges(goal_template, scene_nodes)
        if hyperedges:
            print(f"        [HyperGraph] Generated {len(hyperedges)} hyperedge(s): {[h.name for h in hyperedges]}")
        
        # Step 3: 构建因子图
        scene_edges = scene_graph.get("edges", [])
        factor_graph = self.build_factor_graph(
            scene_nodes, scene_edges, goal_template, hyperedges, node_confidences
        )
        print(f"        [HyperGraph] Factor graph: {len(factor_graph.variable_nodes)} vars, "
              f"{len(factor_graph.unary_factors)} unary, {len(factor_graph.binary_factors)} binary, "
              f"{len(factor_graph.hyper_factors)} hyper factors")
        
        # Step 4: 信念传播推断
        bp = BeliefPropagation(
            factor_graph,
            max_iterations=self.config['bp_max_iterations'],
            damping=self.config['bp_damping']
        )
        raw_prob_map = bp.run()
        raw_top = sorted(raw_prob_map.items(), key=lambda x: -x[1])[:5]
        
        # 过滤低概率节点
        threshold = float(self.config.get('prob_threshold', 0.1))
        prob_map = {k: v for k, v in raw_prob_map.items() if v > threshold}
        if not prob_map and raw_top:
            keep_topk = int(self.config.get('prob_keep_topk', 3))
            keep_floor = max(0.05, threshold * 0.5)
            prob_map = {k: v for k, v in raw_top[:keep_topk] if v > keep_floor}
        
        print(f"        [HyperGraph] Raw top nodes: {raw_top[:3]}")
        print(f"        [HyperGraph] Inference complete. Top nodes: "
              f"{sorted(prob_map.items(), key=lambda x: -x[1])[:3]}")
        
        return prob_map, goal_template
    
    def get_overlap_score(self, prob_map: Dict[str, float], 
                          goal_template: dict) -> float:
        """计算整体重叠分数（兼容原有 overlap() 接口）
        
        将 prob_map 转换为单一的重叠分数 [0, 1]，
        用于与原有探索策略兼容。
        """
        if not prob_map:
            return 0.0
            
        target = goal_template.get("target", "")
        context = goal_template.get("context", [])
        target_tokens = self._caption_tokens(target)

        def _caption_word_stems(node_id: str) -> set:
            """从 UID 中提取 caption 并返回其词级别词干集合。
            UID 格式为 "{caption}_{id(node)}"，用 rsplit 切掉最后一段数字后缀。"""
            caption = node_id.rsplit('_', 1)[0] if '_' in node_id else node_id
            return self._caption_tokens(caption)

        # 目标匹配分数：词级别词干匹配，避免 "table" 命中 "coffee_table_uid"
        # 同时支持多词 caption（如 "wooden bed" → {"wooden","bed"} → 命中 "bed"）
        target_score = 0.0
        for node_id, prob in prob_map.items():
            if target_tokens & _caption_word_stems(node_id):
                target_score = max(target_score, prob)

        # 上下文匹配分数
        context_scores = []
        for ctx in context:
            ctx_tokens = self._caption_tokens(ctx)
            for node_id, prob in prob_map.items():
                if ctx_tokens & _caption_word_stems(node_id):
                    context_scores.append(prob)
                    break
                    
        context_score = np.mean(context_scores) if context_scores else 0.0
        
        # 综合分数
        overlap_score = 0.6 * target_score + 0.4 * context_score
        return float(np.clip(overlap_score, 0.0, 1.0))
    
    def clear_cache(self):
        """清除缓存（在新 episode 开始时调用）"""
        self._hyperedge_cache.clear()
        self._goal_expand_cache.clear()
