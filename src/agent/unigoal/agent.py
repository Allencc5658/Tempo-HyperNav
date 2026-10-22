import warnings
warnings.filterwarnings('ignore')
import math
import os
import re
import cv2
from PIL import Image
import skimage.morphology
from skimage.draw import line_aa, line
import numpy as np
import torch
from torchvision import transforms

from src.utils.fmm.fmm_planner_policy import FMMPlanner
import src.utils.fmm.pose_utils as pu
from src.utils.visualization.semantic_prediction import SemanticPredMaskRCNN
from src.utils.visualization.visualization import (
    init_vis_image,
    draw_line,
    get_contour_points,
    line_list,
    add_text_list
)
from src.utils.visualization.save import save_video
from src.utils.llm import LLM
from src.utils.depth import habitat_depth_to_cm
from src.utils.text_goal import (
    format_text_goal_description,
    infer_text_goal_category_id,
)

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, match_pair , numpy_image_to_torch



class UniGoal_Agent():
    def __init__(self, args, envs):
        self.args = args
        self.envs = envs
        self.device = args.device

        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])


        self.sem_pred = SemanticPredMaskRCNN(args)
        self.llm = LLM(self.args.base_url, self.args.api_key, self.args.llm_model)

        self.selem = skimage.morphology.disk(3)

        self.rgbd = None
        self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None
        self.instance_imagegoal = None
        self.text_goal = None

        self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features='disk').eval().to(self.device)

        self.global_width = args.global_width
        self.global_height = args.global_height
        self.local_width = args.local_width
        self.local_height = args.local_height
        
        self.global_goal = None
        self.global_goal_blocked_count = 0
        self.global_goal_requires_verification = False
        # define a temporal goal with a living time
        self.temp_goal = None
        self.last_temp_goal = None # avoid choose one goal twice
        self.forbidden_temp_goal = []
        self.flag = 0
        self.goal_instance_whwh = None
        # define untraversible area of the goal: 0 means area can be goals, 1 means cannot be
        self.goal_map_mask = np.ones((self.global_width, self.global_height))
        self.pred_box = []
        self.current_target_evidence = {}
        self.last_arrival_verdict = {}
        self.no_detection_verification_count = 0
        self.no_detection_verification_pos = None
        self.arrival_pending_count = 0
        self.arrival_pending_pos = None
        self.prompt_text2object = '"chair: 0, sofa: 1, plant: 2, bed: 3, toilet: 4, tv_monitor: 5" The above are the labels corresponding to each category. Which object is described in the following text? Only response the number of the label and not include other text.\nText: {text}'
        torch.set_grad_enabled(False)

        if args.visualize:
            self.vis_image_background = None
            self.rgb_vis = None
            self.vis_image_list = []

    def reset(self):
        args = self.args

        obs, info = self.envs.reset()

        if self.args.goal_type == 'ins-image':
            self.instance_imagegoal = self.envs.instance_imagegoal
        elif self.args.goal_type == 'text':
            self.text_goal = self.envs.text_goal
        elif self.args.goal_type == 'goat':
            self.instance_imagegoal = self.envs.instance_imagegoal
            self.text_goal = self.envs.text_goal
        idx = self.get_goal_cat_id()
        if idx is not None:
            self.envs.set_goal_cat_id(idx)

        rgbd = np.concatenate((obs['rgb'].astype(np.uint8), obs['depth']), axis=2).transpose(2, 0, 1)
        self.raw_obs = rgbd[:3, :, :].transpose(1, 2, 0)
        self.raw_depth = rgbd[3:4, :, :]

        rgbd, seg_predictions = self.preprocess_obs(rgbd)
        self.rgbd = rgbd

        self.obs_shape = rgbd.shape

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None
        self.global_goal = None
        self.global_goal_blocked_count = 0
        self.temp_goal = None
        self.last_temp_goal = None
        self.forbidden_temp_goal = []
        self.goal_map_mask = np.ones(map_shape)
        self.goal_instance_whwh = None
        self.pred_box = []
        self.been_stuck = False
        self.stuck_goal = None
        self.collision_streak = 0
        self.frontier_vis = None
        self.verification_count = 0
        self.verification_pos = None
        self.verification_steps = []
        self.rejected_goals = []
        self.global_goal_requires_verification = False
        self.depth_debug_count = 0
        self.current_target_evidence = {}
        self.last_arrival_verdict = {}
        self.no_detection_verification_count = 0
        self.no_detection_verification_pos = None
        self.arrival_pending_count = 0
        self.arrival_pending_pos = None

        if args.visualize:
            self.vis_image_background = init_vis_image(self.envs.goal_name, self.args)

        return obs, rgbd, info

    def update_goat_goal(self, infos):
        """Update goal-related state for a GOAT subtask switch (no env/map reset)."""
        from src.envs.goat_episode_parser import GOAL_TYPE_IMAGE, GOAL_TYPE_DESCRIPTION
        goal_type = infos.get('goal_type', 'object')
        self.instance_imagegoal = None
        self.text_goal = None
        if goal_type == GOAL_TYPE_IMAGE:
            self.instance_imagegoal = self.envs.instance_imagegoal
        elif goal_type == GOAL_TYPE_DESCRIPTION:
            self.text_goal = self.envs.text_goal
        # Reset goal-search state so the agent hunts for the new target
        map_shape = (self.args.map_size_cm // self.args.map_resolution,
                     self.args.map_size_cm // self.args.map_resolution)
        self.global_goal = None
        self.global_goal_blocked_count = 0
        self.temp_goal = None
        self.last_temp_goal = None
        self.forbidden_temp_goal = []
        self.goal_map_mask = np.ones(map_shape)
        self.goal_instance_whwh = None
        self.pred_box = []
        self.been_stuck = False
        self.stuck_goal = None
        self.collision_streak = 0
        self.verification_count = 0
        self.verification_pos = None
        self.verification_steps = []
        self.rejected_goals = []
        self.global_goal_requires_verification = False
        self.depth_debug_count = 0
        self.current_target_evidence = {}
        self.last_arrival_verdict = {}
        self.no_detection_verification_count = 0
        self.no_detection_verification_pos = None
        self.arrival_pending_count = 0
        self.arrival_pending_pos = None

    def _is_instance_image_goal(self):
        return self.args.goal_type == 'ins-image' or (
            self.args.goal_type == 'goat'
            and getattr(self.envs, 'goal_type', None) == 'image'
            and self.instance_imagegoal is not None
        )

    def _target_evidence_from_boxes(self, boxes):
        frame_h = int(getattr(self.args, 'env_frame_height', getattr(self.args, 'frame_height', 1)))
        frame_w = int(getattr(self.args, 'env_frame_width', getattr(self.args, 'frame_width', 1)))
        raw_obs = getattr(self, 'raw_obs', None)
        if raw_obs is not None and raw_obs.ndim >= 2:
            frame_h, frame_w = int(raw_obs.shape[0]), int(raw_obs.shape[1])
        frame_area = max(float(frame_h * frame_w), 1.0)
        areas = []
        max_box = None
        for item in ([] if boxes is None else boxes):
            if len(item) < 3:
                continue
            whwh = np.asarray(item[2], dtype=float).reshape(-1)
            if whwh.size < 4 or not np.all(np.isfinite(whwh[:4])):
                continue
            x1, y1, x2, y2 = whwh[:4]
            x1 = float(np.clip(x1, 0, frame_w))
            x2 = float(np.clip(x2, 0, frame_w))
            y1 = float(np.clip(y1, 0, frame_h))
            y2 = float(np.clip(y2, 0, frame_h))
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area <= 0:
                continue
            areas.append(area)
            if max_box is None or area > max_box[0]:
                max_box = (area, [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])
        evidence = {
            'count': int(len(areas)),
            'detected': bool(areas),
            'max_area_ratio': float(max(areas) / frame_area) if areas else 0.0,
            'total_area_ratio': float(sum(areas) / frame_area) if areas else 0.0,
            'goal_cat_id': int(getattr(self.envs, 'gt_goal_idx', -1)),
        }
        if max_box is not None:
            evidence['max_box'] = max_box[1]
        self.current_target_evidence = evidence
        return evidence

    def _record_arrival_verdict(self, verdict, reason, goal_dis_cm=None, ctx_score=None, target_evidence=None):
        self.last_arrival_verdict = {
            'verdict': verdict,
            'reason': reason,
            'step': int(getattr(self.envs, 'timestep', -1)),
            'goal_dis_cm': None if goal_dis_cm is None else float(goal_dis_cm),
            'ctx_score': None if ctx_score is None else float(ctx_score),
            'target_evidence': dict(target_evidence or getattr(self, 'current_target_evidence', {}) or {}),
        }
        return verdict

    def _target_detection_is_strong(self, target_evidence, goal_dis_cm=None):
        if int(target_evidence.get('count', 0) or 0) <= 0:
            return False
        max_area = float(target_evidence.get('max_area_ratio', 0.0) or 0.0)
        total_area = float(target_evidence.get('total_area_ratio', 0.0) or 0.0)
        strong_area = float(getattr(self.args, 'objnav_strong_detection_area', 0.02))
        near_area = float(getattr(self.args, 'objnav_near_detection_area', 0.006))
        near_cm = float(getattr(self.args, 'objnav_direct_stop_cm', 120.0))
        if max_area >= strong_area or total_area >= strong_area:
            return True
        return goal_dis_cm is not None and goal_dis_cm <= near_cm and max_area >= near_area

    def _update_arrival_pending(self, global_pos):
        if (
            self.arrival_pending_pos is not None
            and np.linalg.norm(global_pos - self.arrival_pending_pos) < 30
        ):
            self.arrival_pending_count += 1
        else:
            self.arrival_pending_count = 1
            self.arrival_pending_pos = global_pos
        return self.arrival_pending_count

    def _reset_arrival_pending(self):
        self.arrival_pending_count = 0
        self.arrival_pending_pos = None

    def _goal_distance_cm_from_map(self, goal_map, start):
        if goal_map is None or not np.any(goal_map > 0):
            return None
        try:
            return float(pu.get_l2_dis_point_map(start, goal_map) * self.args.map_resolution)
        except Exception:
            return None

    def local_feature_match_lightglue(self, re_key2=False):
        with torch.set_grad_enabled(False):
            ob = numpy_image_to_torch(self.raw_obs[:, :, :3]).to(self.device)
            gi = numpy_image_to_torch(self.instance_imagegoal).to(self.device)
            try:
                feats0, feats1, matches01  = match_pair(self.extractor, self.matcher, ob, gi
                    )
                # indices with shape (K, 2)
                matches = matches01['matches']
                # in case that the matches collapse make a check
                b = torch.nonzero(matches[..., 0] < 2048, as_tuple=False)
                c = torch.index_select(matches[..., 0], dim=0, index=b.squeeze())
                points0 = feats0['keypoints'][c]
                if re_key2:
                    return (points0.numpy(), feats1['keypoints'][c].numpy())
                else:
                    return points0.numpy()  
            except:
                if re_key2:
                    # print(f'{self.env.rank}  {self.env.timestep}  h')
                    return (np.zeros((1, 2)), np.zeros((1, 2)))
                else:
                    # print(f'{self.env.rank}  {self.env.timestep}  h')
                    return np.zeros((1, 2))
                
    def compute_ins_dis_v1(self, depth, whwh, k=3):
        '''
        analyze the maxium depth points's pos
        make sure the object is within the range of 10m
        '''
        hist, bins = np.histogram(depth[whwh[1]:whwh[3], whwh[0]:whwh[2]].flatten(), \
            bins=200,range=(0,2000))
        peak_indices = np.argsort(hist)[-k:]  # Get the indices of the top k peaks
        peak_values = hist[peak_indices] + hist[np.clip(peak_indices-1, 0, len(hist)-1)]  + \
            hist[np.clip(peak_indices+1, 0, len(hist)-1)]
        max_area_index = np.argmax(peak_values)  # Find the index of the peak with the largest area
        max_index = peak_indices[max_area_index]
        # max_index = np.argmax(hist)
        return bins[max_index]

    def compute_ins_goal_map(self, whwh, start, start_o):
        goal_mask = np.zeros_like(self.rgbd[3, :, :])
        goal_mask[whwh[1]:whwh[3], whwh[0]:whwh[2]] = 1
        semantic_mask = (self.rgbd[4+self.envs.gt_goal_idx, :, :] > 0) & (goal_mask > 0)

        depth_h, depth_w = np.where(semantic_mask > 0)
        goal_dis = self.rgbd[3, :, :][depth_h, depth_w] / self.args.map_resolution

        goal_angle = -self.args.hfov / 2 * (depth_w - self.rgbd.shape[2]/2) \
        / (self.rgbd.shape[2]/2)
        goal = [start[0]+goal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
            start[1]+goal_dis*np.cos(np.deg2rad(start_o+goal_angle))]
        goal_map = np.zeros((self.local_width, self.local_height))
        goal[0] = np.clip(goal[0], 0, self.local_width - 1).astype(int)
        goal[1] = np.clip(goal[1], 0, self.local_height - 1).astype(int)
        goal_map[goal[0], goal[1]] = 1
        return goal_map

    def _set_square(self, mat, center, radius, value):
        row, col = int(center[0]), int(center[1])
        r1, r2 = max(0, row - radius), min(mat.shape[0], row + radius + 1)
        c1, c2 = max(0, col - radius), min(mat.shape[1], col + radius + 1)
        mat[r1:r2, c1:c2] = value

    def _global_goal_to_local_waypoint(self, global_goal, gx1, gx2, gy1, gy2):
        local_map = np.zeros((gx2 - gx1, gy2 - gy1))
        rows, cols = np.where(global_goal > 0)
        if len(rows) == 0:
            return local_map
        target_row = float(np.mean(rows))
        target_col = float(np.mean(cols))
        local_row = int(np.clip(round(target_row - gx1), 0, local_map.shape[0] - 1))
        local_col = int(np.clip(round(target_col - gy1), 0, local_map.shape[1] - 1))
        local_map[local_row, local_col] = 1
        return local_map

    def _local_goal_or_waypoint(self, global_goal, gx1, gx2, gy1, gy2):
        local_goal = pu.threshold_pose_map(global_goal, gx1, gx2, gy1, gy2, clip=False)
        if np.any(local_goal > 0):
            return local_goal, True
        waypoint = self._global_goal_to_local_waypoint(global_goal, gx1, gx2, gy1, gy2)
        return waypoint, False

    def _select_escape_goal(self, map_pred, start, gx1, gx2, gy1, gy2):
        local_collision = self.collision_map[gx1:gx2, gy1:gy2] == 1
        local_free = (np.rint(map_pred) == 0) & (~local_collision)
        if not np.any(local_free):
            return np.asarray(start, dtype=int)

        free_uint = local_free.astype(np.uint8)
        clearance = cv2.distanceTransform(free_uint, cv2.DIST_L2, 5)
        rows, cols = np.where(local_free)
        points = np.stack([rows, cols], axis=1)
        dists = np.linalg.norm(points - np.asarray(start), axis=1)

        valid = dists >= 8
        if np.any(valid):
            points = points[valid]
            rows = rows[valid]
            cols = cols[valid]
            dists = dists[valid]

        local_visited = self.visited[gx1:gx2, gy1:gy2]
        visit_penalty = np.clip(local_visited[rows, cols], 0, 5) * 0.2
        target_dist = 40.0
        scores = (
            2.0 * clearance[rows, cols]
            + 0.05 * dists
            - 0.01 * np.abs(dists - target_dist)
            - visit_penalty
        )
        selected = int(np.argmax(scores))
        goal = points[selected]
        return pu.threshold_poses(goal, map_pred.shape)

    def _is_near_rejected(self, goal_map, gx1, gx2, gy1, gy2, threshold=40):
        """检查候选目标是否靠近已拒绝的目标位置"""
        if not self.rejected_goals:
            return False
        goal_pos = np.array(np.where(goal_map > 0)).mean(axis=1)  # local (r, c)
        global_pos = np.array([goal_pos[0] + gx1, goal_pos[1] + gy1])
        for rg in self.rejected_goals:
            if np.linalg.norm(global_pos - np.array(rg)) < threshold:
                return True
        return False

    def _context_match_score(self, goal_map, planner_inputs, gx1, gx2, gy1, gy2):
        """计算候选目标周围的场景图节点与目标图上下文的匹配度
        
        返回 [0, 1]：0 = 完全不匹配，1 = 完美匹配
        """
        scene_nodes = planner_inputs.get('scene_nodes', [])
        goalgraph = planner_inputs.get('goalgraph', None)
        if not scene_nodes or not goalgraph:
            return 0.0  # 信息不足时保守拒绝，避免误判到达

        # 从目标图提取上下文物体名（排除目标本身）
        goal_name = self.envs.goal_name.lower() if hasattr(self.envs, 'goal_name') else ""
        context_names = set()
        for node in goalgraph.get('nodes', []):
            name = node.get('id', '').lower().strip()
            if name and name != goal_name:
                context_names.add(name)
        if not context_names:
            return 0.0  # 目标图无上下文节点（LLM 解析可能失败），保守拒绝

        # 获取候选目标在全局地图中的位置
        goal_pos = np.array(np.where(goal_map > 0)).mean(axis=1)
        global_goal = np.array([goal_pos[0] + gx1, goal_pos[1] + gy1])

        # 统计候选目标周围 150 格（~7.5m）内的场景节点
        nearby_captions = set()
        for sn in scene_nodes:
            if sn['center'] is not None:
                dist = np.linalg.norm(np.array(sn['center']) - global_goal)
                if dist < 150:
                    nearby_captions.add(sn['caption'].lower().strip())

        if not nearby_captions:
            return 0.5  # 周围无节点信息，给中间值

        # 计算上下文匹配率
        matched = sum(1 for c in context_names if any(c in nc or nc in c for nc in nearby_captions))
        return matched / len(context_names)

    def _format_depth_stats(self, arr):
        if arr is None:
            return 'none'
        arr = np.asarray(arr, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 'no-finite'
        return (
            f"min={float(np.min(finite)):.3f}, "
            f"p50={float(np.percentile(finite, 50)):.3f}, "
            f"p99={float(np.percentile(finite, 99)):.3f}, "
            f"max={float(np.max(finite)):.3f}"
        )

    def _log_depth_debug(self, tag, goal_dis_cm=None, ctx_score=None, extra=''):
        if getattr(self, 'depth_debug_count', 0) >= 20:
            return
        self.depth_debug_count = getattr(self, 'depth_debug_count', 0) + 1
        raw_depth = getattr(self, 'raw_depth', None)
        proc_depth = self.rgbd[3, :, :] if self.rgbd is not None and self.rgbd.shape[0] > 3 else None
        if raw_depth is not None and raw_depth.ndim == 3:
            raw_depth = raw_depth[0]
        far_ratio = 'none'
        if raw_depth is not None:
            finite_raw = np.asarray(raw_depth)[np.isfinite(raw_depth)]
            if finite_raw.size > 0:
                far_ratio = f"{float(np.mean(finite_raw >= 0.99)):.3f}"
        step = getattr(self.envs, 'timestep', -1)
        goal_name = getattr(self.envs, 'goal_name', '')
        goal_dis_text = 'none' if goal_dis_cm is None else f"{float(goal_dis_cm):.1f}cm"
        ctx_text = 'none' if ctx_score is None else f"{float(ctx_score):.2f}"
        print(
            f"[{tag}] step={step}, goal={goal_name}, goal_dis={goal_dis_text}, "
            f"ctx={ctx_text}, raw_norm=({self._format_depth_stats(raw_depth)}), "
            f"raw_far_ratio={far_ratio}, proc_cm=({self._format_depth_stats(proc_depth)}) {extra}"
        )

    def verify_arrival(self, goal_map, planner_inputs, gx1, gx2, gy1, gy2):
        """到达验证器：对 text/objnav 模式的候选目标进行多维度确认
        
        验证逻辑：
        1. 检查候选目标是否靠近已被拒绝的目标 → 直接拒绝
        2. 上下文匹配：检查周围场景节点是否包含目标图中的上下文物体
        3. 多次确认：要求从不同视角至少确认 2 次
        
        Returns:
            'confirmed': 确认到达
            'pending': 继续接近观察
            'rejected': 拒绝该候选，标记为非目标
        """
        # 检查0：前 MIN_VERIFY_STEP 步场景图尚未充分建立，不允许确认
        MIN_VERIFY_STEP = int(getattr(self.args, 'min_arrival_verify_step', 30))
        if hasattr(self.envs, 'timestep') and self.envs.timestep < MIN_VERIFY_STEP:
            self._log_depth_debug('ArrivalVerifier', extra=f'too early (<{MIN_VERIFY_STEP})')
            print(f"  [ArrivalVerifier] Too early (step={self.envs.timestep}) → pending")
            return self._record_arrival_verdict('pending', 'too_early')

        # 检查1：是否靠近已拒绝目标
        if self._is_near_rejected(goal_map, gx1, gx2, gy1, gy2):
            print(f"  [ArrivalVerifier] Candidate near rejected location → rejected")
            return self._record_arrival_verdict('rejected', 'near_rejected')

        # 检查2：上下文匹配
        ctx_score = self._context_match_score(goal_map, planner_inputs, gx1, gx2, gy1, gy2)
        
        # 上下文严重不匹配 → 直接拒绝
        if ctx_score < 0.2 and len(self.rejected_goals) == 0:
            # 仅在有足够上下文信息时（score < 0.2 意味着上下文存在但不匹配）才拒绝
            goalgraph = planner_inputs.get('goalgraph', None)
            if goalgraph and len(goalgraph.get('nodes', [])) > 1:
                print(f"  [ArrivalVerifier] Context mismatch (score={ctx_score:.2f}) → rejected")
                return self._record_arrival_verdict('rejected', 'context_mismatch', ctx_score=ctx_score)

        # 检查3：多次确认
        goal_pos = np.array(np.where(goal_map > 0)).mean(axis=1)
        global_pos = np.array([goal_pos[0] + gx1, goal_pos[1] + gy1])
        start_x, start_y, _, px1, _, py1, _ = planner_inputs['pose_pred']
        start = np.array([
            int(start_y * 100.0 / self.args.map_resolution - int(px1)),
            int(start_x * 100.0 / self.args.map_resolution - int(py1)),
        ])
        goal_dis_cm = float(np.linalg.norm(goal_pos - start) * self.args.map_resolution)
        target_evidence = planner_inputs.get('target_evidence') or getattr(self, 'current_target_evidence', {}) or {}
        target_count = int(target_evidence.get('count', 0) or 0)
        target_area = float(target_evidence.get('max_area_ratio', 0.0) or 0.0)
        self._log_depth_debug(
            'ArrivalVerifier',
            goal_dis_cm=goal_dis_cm,
            ctx_score=ctx_score,
            extra=f'target_count={target_count}, target_area={target_area:.4f}',
        )

        if self.args.goal_type == 'objnav' and target_count <= 0:
            pending_count = self._update_arrival_pending(global_pos)
            pending_reject_count = int(getattr(self.args, 'objnav_pending_reject_count', 12))
            if pending_count >= pending_reject_count:
                print(
                    "  [ArrivalVerifier] ObjNav candidate stayed pending without "
                    f"target detection ({pending_count}/{pending_reject_count}) -> rejected"
                )
                return self._record_arrival_verdict(
                    'rejected',
                    'objnav_pending_no_detection',
                    goal_dis_cm=goal_dis_cm,
                    ctx_score=ctx_score,
                    target_evidence=target_evidence,
                )
            if (
                self.no_detection_verification_pos is not None
                and np.linalg.norm(global_pos - self.no_detection_verification_pos) < 30
            ):
                self.no_detection_verification_count += 1
            else:
                self.no_detection_verification_count = 1
                self.no_detection_verification_pos = global_pos
            reject_count = int(getattr(self.args, 'objnav_arrival_reject_count', 2))
            if self.no_detection_verification_count >= reject_count:
                print(
                    "  [ArrivalVerifier] ObjNav close candidate has no target "
                    f"detection ({self.no_detection_verification_count}/{reject_count}) -> rejected"
                )
                return self._record_arrival_verdict(
                    'rejected',
                    'objnav_no_detection',
                    goal_dis_cm=goal_dis_cm,
                    ctx_score=ctx_score,
                    target_evidence=target_evidence,
                )
            print(
                "  [ArrivalVerifier] ObjNav close candidate has no target "
                f"detection ({self.no_detection_verification_count}/{reject_count}) -> pending"
            )
            return self._record_arrival_verdict(
                'pending',
                'objnav_no_detection',
                goal_dis_cm=goal_dis_cm,
                ctx_score=ctx_score,
                target_evidence=target_evidence,
            )
        self.no_detection_verification_count = 0
        self.no_detection_verification_pos = None

        if self.args.goal_type == 'objnav' and self._target_detection_is_strong(target_evidence, goal_dis_cm):
            self.verification_count = 0
            self.verification_pos = None
            self.verification_steps = []
            self._reset_arrival_pending()
            print(
                "  [ArrivalVerifier] ObjNav strong target evidence "
                f"(count={target_count}, area={target_area:.4f}, dist={goal_dis_cm:.1f}cm) -> confirmed"
            )
            return self._record_arrival_verdict(
                'confirmed',
                'objnav_strong_detection',
                goal_dis_cm=goal_dis_cm,
                ctx_score=ctx_score,
                target_evidence=target_evidence,
            )
        
        if self.verification_pos is not None:
            if np.linalg.norm(global_pos - self.verification_pos) < 30:
                self.verification_count += 1
            else:
                # 新位置的候选目标，重置计数
                self.verification_count = 1
                self.verification_pos = global_pos
        else:
            self.verification_count = 1
            self.verification_pos = global_pos

        now_step = int(getattr(self.envs, 'timestep', -1))
        min_gap = int(getattr(self.args, 'arrival_verify_gap', 2))
        if self.verification_count <= 1:
            self.verification_steps = []
        last_step = self.verification_steps[-1] if self.verification_steps else None
        if last_step is None or now_step - last_step >= min_gap:
            self.verification_steps.append(now_step)
        self.verification_count = len(self.verification_steps)

        required_count = int(getattr(self.args, 'arrival_verify_count', 3))
        if ctx_score >= 0.8:
            required_count = max(2, required_count - 1)
        if self.args.goal_type == 'objnav' and target_count > 0:
            required_count = min(required_count, 2)

        if self.verification_count >= required_count:
            print(f"  [ArrivalVerifier] Confirmed (count={self.verification_count}, ctx={ctx_score:.2f})")
            self.verification_count = 0
            self.verification_pos = None
            self.verification_steps = []
            self._reset_arrival_pending()
            return self._record_arrival_verdict(
                'confirmed',
                'verified',
                goal_dis_cm=goal_dis_cm,
                ctx_score=ctx_score,
                target_evidence=target_evidence,
            )
        else:
            pending_count = self._update_arrival_pending(global_pos)
            pending_reject_count = int(getattr(self.args, 'objnav_pending_reject_count', 12))
            if self.args.goal_type == 'objnav' and pending_count >= pending_reject_count:
                print(
                    "  [ArrivalVerifier] ObjNav candidate stayed pending "
                    f"({pending_count}/{pending_reject_count}) -> rejected"
                )
                return self._record_arrival_verdict(
                    'rejected',
                    'objnav_pending_timeout',
                    goal_dis_cm=goal_dis_cm,
                    ctx_score=ctx_score,
                    target_evidence=target_evidence,
                )
            print(f"  [ArrivalVerifier] Pending (count={self.verification_count}/{required_count}, ctx={ctx_score:.2f})")
            return self._record_arrival_verdict(
                'pending',
                'verify_count',
                goal_dis_cm=goal_dis_cm,
                ctx_score=ctx_score,
                target_evidence=target_evidence,
            )

    def _reject_goal_location(self, goal_map, gx1, gx2, gy1, gy2):
        """将候选目标位置标记为已拒绝"""
        goal_pos = np.array(np.where(goal_map > 0)).mean(axis=1)
        global_pos = [goal_pos[0] + gx1, goal_pos[1] + gy1]
        self.rejected_goals.append(global_pos)
        # 标记 goal_map_mask 禁止该区域
        selem = skimage.morphology.disk(5)
        dilated = skimage.morphology.dilation(goal_map, selem)
        self.goal_map_mask[gx1:gx2, gy1:gy2][dilated > 0] = 0
        self.temp_goal = None
        self.verification_count = 0
        self.verification_pos = None
        self.verification_steps = []
        self.no_detection_verification_count = 0
        self.no_detection_verification_pos = None
        self._reset_arrival_pending()
        print(f"  [ArrivalVerifier] Location rejected and masked. Total rejected: {len(self.rejected_goals)}")

    def instance_discriminator(self, planner_inputs, id_lo_whwh_speci):
        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        map_pred = np.rint(planner_inputs['map_pred'])
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        r, c = start_y, start_x
        start = [int(r * 100.0 / self.args.map_resolution - gx1),
                 int(c * 100.0 / self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        if self.instance_imagegoal is None and self.text_goal is None and self.args.goal_type not in ('objnav', 'goat'):
            # not initialized (objnav/goat uses category detection directly)
            return planner_inputs
        elif self.been_stuck:
            planner_inputs['found_goal'] = 0
            if self.stuck_goal is None:
                goal = self._select_escape_goal(map_pred, start, gx1, gx2, gy1, gy2)
                self.stuck_goal = [int(goal[0])+gx1, int(goal[1])+gy1]
            else:
                goal = np.array([self.stuck_goal[0]-gx1, self.stuck_goal[1]-gy1])
                goal = pu.threshold_poses(goal, map_pred.shape)
            planner_inputs['goal'] = np.zeros((self.local_width, self.local_height))
            planner_inputs['goal'][int(goal[0]), int(goal[1])] = 1
            return planner_inputs
        elif self.global_goal is not None:
            goal_map, goal_in_window = self._local_goal_or_waypoint(self.global_goal, gx1, gx2, gy1, gy2)
            planner_inputs['goal'] = goal_map if np.any(goal_map > 0) else planner_inputs['exp_goal']
            planner_inputs['found_goal'] = 0
            if (
                goal_in_window
                and getattr(self, 'global_goal_requires_verification', False)
                and self.args.goal_type in ('text', 'objnav', 'goat')
            ):
                goal_dis = self.compute_temp_goal_distance(
                    map_pred, goal_map, start, planning_window
                )
                if goal_dis is not None and goal_dis < 15:
                    verdict = self.verify_arrival(goal_map, planner_inputs, gx1, gx2, gy1, gy2)
                    if verdict == 'confirmed':
                        planner_inputs['found_goal'] = 1
                        self.global_goal_requires_verification = False
                    elif verdict == 'rejected':
                        self._reject_goal_location(goal_map, gx1, gx2, gy1, gy2)
                        self.global_goal = None
                        planner_inputs['goal'] = planner_inputs['exp_goal']
                    # pending keeps navigating, but cannot trigger STOP.
            elif goal_in_window:
                planner_inputs['found_goal'] = 1
            return planner_inputs
        elif planner_inputs['found_goal'] == 1:
            goal_channel = 4 + self.envs.gt_goal_idx
            if not 0 <= goal_channel < self.rgbd.shape[0]:
                planner_inputs['found_goal'] = 0
                planner_inputs['goal'] = planner_inputs['exp_goal']
                return planner_inputs
            goal_mask = self.rgbd[goal_channel, :, :]
            id_lo_whwh_speci = sorted(id_lo_whwh_speci, 
                key=lambda s: (s[2][2]-s[2][0])**2+(s[2][3]-s[2][1])**2, reverse=True)
            whwh = (id_lo_whwh_speci[0][2] / 4).astype(int)
            w, h = whwh[2]-whwh[0], whwh[3]-whwh[1]
            goal_mask = np.zeros_like(goal_mask)
            goal_mask[whwh[1]:whwh[3], whwh[0]:whwh[2]] = 1.

            if self._is_instance_image_goal():
                index = self.local_feature_match_lightglue()
                match_points = index.shape[0]
            planner_inputs['found_goal'] = 0

            if self.temp_goal is not None:
                goal_map, temp_in_window = self._local_goal_or_waypoint(self.temp_goal, gx1, gx2, gy1, gy2)
                if not temp_in_window:
                    planner_inputs['found_goal'] = 0
                    planner_inputs['goal'] = goal_map if np.any(goal_map > 0) else planner_inputs['exp_goal']
                    return planner_inputs
                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)
            else:
                goal_map = self.compute_ins_goal_map(whwh, start, start_o)
                if not np.any(goal_map>0) :
                    tgoal_dis = self.compute_ins_dis_v1(self.rgbd[3, :, :], whwh) / self.args.map_resolution
                    rgb_center = np.array([whwh[3]+whwh[1], whwh[2]+whwh[0]])//2
                    goal_angle = -self.args.hfov / 2 * (rgb_center[1] - self.rgbd.shape[2]/2) \
                    / (self.rgbd.shape[2]/2)
                    goal = [start[0]+tgoal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
                        start[1]+tgoal_dis*np.cos(np.deg2rad(start_o+goal_angle))]
                    goal = pu.threshold_poses(goal, map_pred.shape)
                    rr,cc = skimage.draw.ellipse(goal[0], goal[1], 10, 10, shape=goal_map.shape)
                    goal_map[rr, cc] = 1


                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)

            if goal_dis is None:
                self.temp_goal = None
                planner_inputs['goal'] = planner_inputs['exp_goal']
                selem = skimage.morphology.disk(3)
                goal_map = skimage.morphology.dilation(goal_map, selem)
                self.goal_map_mask[gx1:gx2, gy1:gy2][goal_map > 0] = 0
                print(f"Rank: {getattr(self.envs, 'rank', 0)}, timestep: {self.envs.timestep},  temp goal unavigable !")
            else:
                if self._is_instance_image_goal() and match_points > 100:
                    planner_inputs['found_goal'] = 1
                    global_goal = np.zeros((self.global_width, self.global_height))
                    global_goal[gx1:gx2, gy1:gy2] = goal_map
                    self.global_goal = global_goal
                    self.global_goal_blocked_count = 0
                    self.global_goal_requires_verification = False
                    planner_inputs['goal'] = goal_map
                    self.temp_goal = None
                else:
                    if (self._is_instance_image_goal() and goal_dis < 50) or (not self._is_instance_image_goal() and self.args.goal_type in ('text', 'objnav', 'goat') and goal_dis < 15):
                        if self._is_instance_image_goal() and match_points > 90:
                            planner_inputs['found_goal'] = 1
                            global_goal = np.zeros((self.global_width, self.global_height))
                            global_goal[gx1:gx2, gy1:gy2] = goal_map
                            self.global_goal = global_goal
                            self.global_goal_blocked_count = 0
                            self.global_goal_requires_verification = False
                            planner_inputs['goal'] = goal_map
                            self.temp_goal = None
                        elif not self._is_instance_image_goal() and self.args.goal_type in ('text', 'objnav', 'goat'):
                            verdict = self.verify_arrival(goal_map, planner_inputs, gx1, gx2, gy1, gy2)
                            if verdict == 'confirmed':
                                planner_inputs['found_goal'] = 1
                                global_goal = np.zeros((self.global_width, self.global_height))
                                global_goal[gx1:gx2, gy1:gy2] = goal_map
                                self.global_goal = global_goal
                                self.global_goal_blocked_count = 0
                                self.global_goal_requires_verification = False
                                planner_inputs['goal'] = goal_map
                                self.temp_goal = None
                            elif verdict == 'rejected':
                                self._reject_goal_location(goal_map, gx1, gx2, gy1, gy2)
                                planner_inputs['found_goal'] = 0
                                planner_inputs['goal'] = planner_inputs['exp_goal']
                            else:  # pending
                                planner_inputs['found_goal'] = 0
                                planner_inputs['goal'] = goal_map
                        else:
                            planner_inputs['goal'] = planner_inputs['exp_goal']
                            self.temp_goal = None
                            selem = skimage.morphology.disk(1)
                            goal_map = skimage.morphology.dilation(goal_map, selem)
                            self.goal_map_mask[gx1:gx2, gy1:gy2][goal_map > 0] = 0
                    else:
                        new_goal_map = goal_map * self.goal_map_mask[gx1:gx2, gy1:gy2]
                        if np.any(new_goal_map > 0):
                            planner_inputs['goal'] = new_goal_map
                            temp_goal = np.zeros((self.global_width, self.global_height))
                            temp_goal[gx1:gx2, gy1:gy2] = new_goal_map
                            self.temp_goal = temp_goal
                        else:
                            planner_inputs['goal'] = planner_inputs['exp_goal']
                            self.temp_goal = None
            return planner_inputs

        else:
            planner_inputs['goal'] = planner_inputs['exp_goal']
            if self.temp_goal is not None:  
                goal_map, temp_in_window = self._local_goal_or_waypoint(self.temp_goal, gx1, gx2, gy1, gy2)
                if not temp_in_window:
                    planner_inputs['goal'] = goal_map if np.any(goal_map > 0) else planner_inputs['exp_goal']
                    planner_inputs['found_goal'] = 0
                    return planner_inputs
                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)
                planner_inputs['found_goal'] = 0
                new_goal_map = goal_map * self.goal_map_mask[gx1:gx2, gy1:gy2]
                if np.any(new_goal_map > 0):
                    if goal_dis is not None:
                        planner_inputs['goal'] = new_goal_map
                        if goal_dis < 100:
                            if self._is_instance_image_goal():
                                index = self.local_feature_match_lightglue()
                                match_points = index.shape[0]
                            if self.args.goal_type == 'objnav':
                                self._record_arrival_verdict(
                                    'rejected',
                                    'objnav_temp_goal_no_detection',
                                    goal_dis_cm=float(goal_dis),
                                    target_evidence=planner_inputs.get('target_evidence'),
                                )
                                self._reject_goal_location(new_goal_map, gx1, gx2, gy1, gy2)
                                planner_inputs['goal'] = planner_inputs['exp_goal']
                                planner_inputs['found_goal'] = 0
                            elif (self._is_instance_image_goal() and match_points < 80) or (not self._is_instance_image_goal() and self.args.goal_type in ('text', 'goat')):
                                planner_inputs['goal'] = planner_inputs['exp_goal']
                                selem = skimage.morphology.disk(3)
                                new_goal_map = skimage.morphology.dilation(new_goal_map, selem)
                                self.goal_map_mask[gx1:gx2, gy1:gy2][new_goal_map > 0] = 0
                                self.temp_goal = None
                    else:
                        selem = skimage.morphology.disk(3)
                        new_goal_map = skimage.morphology.dilation(new_goal_map, selem)
                        self.goal_map_mask[gx1:gx2, gy1:gy2][new_goal_map > 0] = 0
                        self.temp_goal = None
                        print(f"Rank: {getattr(self.envs, 'rank', 0)}, timestep: {self.envs.timestep},  temp goal unavigable !")
                else:
                    self.temp_goal = None
                    
                    
            return planner_inputs


    def step(self, agent_input):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        # plan
        if agent_input["wait"]:
            self.last_action = None
            self.envs.info["sensor_pose"] = [0., 0., 0.]
            return None, np.zeros(self.rgbd.shape), False, self.envs.info

        id_lo_whwh = self.pred_box

        semantic_direct_goal = agent_input.get('semantic_direct_goal')
        if semantic_direct_goal is not None:
            goal_rc = np.rint(np.asarray(semantic_direct_goal)[:2]).astype(int)
            if (
                0 <= goal_rc[0] < self.global_width
                and 0 <= goal_rc[1] < self.global_height
            ):
                self.global_goal = np.zeros((self.global_width, self.global_height))
                self.global_goal[goal_rc[0], goal_rc[1]] = 1.0
                self.global_goal_blocked_count = 0
                self.global_goal_requires_verification = (
                    self.args.goal_type in ('text', 'objnav', 'goat')
                )

        if self.args.goal_type == 'goat' and self.envs.gt_goal_idx == -1:
            # Unknown Mask R-CNN labels are handled by the labelled
            # GroundingDINO/SAM scene graph. Never treat every COCO box as the
            # target: that makes the first visible object a false positive.
            id_lo_whwh_speci = []
        elif self.envs.gt_goal_idx == -1:
            # Unknown text/GOAT categories should not turn every COCO box into
            # a possible target. Let graph/frontier search handle them.
            if self.args.goal_type in ('text', 'goat'):
                id_lo_whwh_speci = []
            else:
                id_lo_whwh_speci = list(id_lo_whwh)
        else:
            id_lo_whwh_speci = [id_lo_whwh[i] for i in range(len(id_lo_whwh)) \
                        if id_lo_whwh[i][0] == self.envs.gt_goal_idx]


        target_evidence = self._target_evidence_from_boxes(id_lo_whwh_speci)
        agent_input["target_evidence"] = target_evidence
        agent_input["found_goal"] = target_evidence['detected']

        self.instance_discriminator(agent_input, id_lo_whwh_speci)

        action = self.get_action(agent_input)

        if self.args.visualize:
            self.visualize(agent_input)

        if action >= 0:
            action = {'action': action}
            obs, done, info = self.envs.step(action)
            rgbd = np.concatenate((obs['rgb'].astype(np.uint8), obs['depth']), axis=2).transpose(2, 0, 1)
            self.raw_obs = rgbd[:3, :, :].transpose(1, 2, 0)
            self.raw_depth = rgbd[3:4, :, :]

            rgbd, seg_predictions = self.preprocess_obs(rgbd) 
            self.last_action = action['action']
            self.rgbd = rgbd

            if done:
                episode_result_info = dict(info)
                goat_dataset_exhausted = (
                    self.args.goal_type == 'goat'
                    and self.envs.goat_episode_idx + 1 >= len(self.envs.goat_parser)
                )
                if not goat_dataset_exhausted:
                    obs, rgbd, info = self.reset()
                self.envs.info['_episode_result_info'] = episode_result_info

            return obs, rgbd, done, self.envs.info

        else:
            self.last_action = None
            self.envs.info["sensor_pose"] = [0., 0., 0.]
            return None, np.zeros(self.obs_shape), False, self.envs.info

    def get_action(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)\
        
        # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                        int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        # self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
        #                                start[1] - 0:start[1] + 1] = 1
        rr, cc, _ = line_aa(last_start[0], last_start[1], start[0], start[1])
        self.visited[gx1:gx2, gy1:gy2][rr, cc] += 1

        if args.visualize:            
            self.visited_vis[gx1:gx2, gy1:gy2] = \
                draw_line(last_start, start,
                             self.visited_vis[gx1:gx2, gy1:gy2])

        # relieve the stuck goal
        x1, y1, t1 = self.last_loc
        x2, y2, _ = self.curr_loc
        if abs(x1 - x2) >= 0.05 or abs(y1 - y2) >= 0.05:
            self.been_stuck = False
            self.stuck_goal = None
            self.collision_streak = 0
            self.global_goal_blocked_count = 0

        # Collision check
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2
            dist = pu.get_l2_distance(x1, x2, y1, y2)
            stuck_motion = dist < args.collision_threshold

            if stuck_motion:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                    self.been_stuck = True
                self.col_width = min(self.col_width, 5)
                self.collision_streak += 1
            else:
                self.collision_streak = 0
                self.col_width = 1

            if stuck_motion:  # Collision
                if self.collision_streak >= int(getattr(self.args, 'stuck_collision_streak', 3)):
                    self.been_stuck = True
                    self.temp_goal = None
                    if self.args.goal_type == 'goat' and self.global_goal is not None:
                        self.global_goal_blocked_count += 1
                        blocked_retries = int(getattr(self.args, 'global_goal_blocked_retries', 1))
                        if self.global_goal_blocked_count >= blocked_retries:
                            print('[GOATStuck] release blocked global_goal and replan')
                            self.global_goal = None
                            self.global_goal_blocked_count = 0
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

        local_goal, stop = self.get_local_goal(map_pred, start, np.copy(goal),
                                  planning_window)

        if self.args.goal_type == 'objnav' and planner_inputs['found_goal'] == 1:
            target_evidence = planner_inputs.get('target_evidence') or getattr(self, 'current_target_evidence', {}) or {}
            goal_dis_cm = self._goal_distance_cm_from_map(goal, start)
            direct_stop_cm = float(getattr(self.args, 'objnav_direct_stop_cm', 120.0))
            if (
                goal_dis_cm is not None
                and goal_dis_cm <= direct_stop_cm
                and self._target_detection_is_strong(target_evidence, goal_dis_cm)
            ):
                stop = True
                self._record_arrival_verdict(
                    'confirmed',
                    'objnav_direct_stop',
                    goal_dis_cm=goal_dis_cm,
                    target_evidence=target_evidence,
                )

        if stop and planner_inputs['found_goal'] == 1 and not self._is_instance_image_goal() and self.args.goal_type in ('text', 'objnav', 'goat'):
            min_stop_step = int(getattr(self.args, 'min_stop_step', 30))
            if getattr(self.envs, 'timestep', 0) < min_stop_step:
                self._log_depth_debug('StopGuard', extra=f'suppress stop before step {min_stop_step}')
                stop = False

        if stop and planner_inputs['found_goal'] == 1:
            self._log_depth_debug('StopAction', extra='send STOP')
            action = 0
        else:
            (local_x, local_y) = local_goal
            angle_st_goal = math.degrees(math.atan2(local_x - start[0],
                                                    local_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.args.turn_angle:
                action = 3
            elif relative_angle < -self.args.turn_angle:
                action = 2
            else:
                action = 1

        return action

    def get_local_goal(self, grid, start, goal, planning_window):
        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        if not np.any(goal > 0):
            goal = np.zeros_like(goal)
            goal[int(start[0]), int(start[1])] = 1

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] > 0] = 1
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        
        self._set_square(traversible, [start[0] - x1, start[1] - y1], 1, 1)

        traversible = self.add_boundary(traversible)
        goal = self.add_boundary(goal, value=0)
        visited = self.add_boundary(self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2], value=0)

        planner = FMMPlanner(traversible)
        if self.stuck_goal is not None:
            selem = skimage.morphology.disk(1)
            goal = skimage.morphology.binary_dilation(
                goal, selem) != True
        elif self.global_goal is not None or self.temp_goal is not None:
            selem = skimage.morphology.disk(10)
            goal = skimage.morphology.binary_dilation(
                goal, selem) != True
        else:
            selem = skimage.morphology.disk(3)
            goal = skimage.morphology.binary_dilation(
                goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]


        if self.global_goal is not None and self.stuck_goal is None:
            st_dis = pu.get_l2_dis_point_map(state, goal) * self.args.map_resolution
            fmm_dist = planner.fmm_dist * self.args.map_resolution 
            dis = fmm_dist[start[0]+1, start[1]+1]
            if st_dis < 100 and dis/st_dis > 2:
                return (0, 0), True

        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)
        if replan:
            stg_x, stg_y, _, stop = planner.get_short_term_goal(state, 2)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop

    def add_boundary(self, mat, value=1):
        h, w = mat.shape
        new_mat = np.zeros((h + 2, w + 2)) + value
        new_mat[1:h + 1, 1:w + 1] = mat
        return new_mat

    def compute_temp_goal_distance(self, grid, goal_map, start, planning_window):
        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape
        goal = goal_map * 1
        if not np.any(goal > 0):
            return None
        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], self.selem)
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] > 0] = 1
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        
        self._set_square(traversible, [start[0] - x1, start[1] - y1], 1, 1)

        st_dis = pu.get_l2_dis_point_map(start, goal) * self.args.map_resolution  # cm

        traversible = self.add_boundary(traversible)
        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        
        goal = cv2.dilate(goal, selem)
        
        goal = self.add_boundary(goal, value=0)
        planner.set_multi_goal(goal)
        fmm_dist = planner.fmm_dist * self.args.map_resolution 
        dis = fmm_dist[start[0]+1, start[1]+1]

        if dis < fmm_dist.max() and dis/st_dis < 2:
            return dis
        else:
            return None

    def preprocess_obs(self, obs, use_seg=True):
        args = self.args
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]

        sem_seg_pred, seg_predictions = self.pred_sem(
            rgb.astype(np.uint8), use_seg=use_seg)

        if args.environment == 'habitat':
            depth = self.preprocess_depth(depth, args.min_depth, args.max_depth)

        width_scale = args.env_frame_width / args.frame_width
        height_scale = args.env_frame_height / args.frame_height
        if (
            not float(width_scale).is_integer()
            or not float(height_scale).is_integer()
            or int(width_scale) != int(height_scale)
        ):
            raise ValueError(
                "Expected equal integer RGB/depth downsample factors, got "
                f"{args.env_frame_width}x{args.env_frame_height} -> "
                f"{args.frame_width}x{args.frame_height}"
            )
        ds = int(width_scale)  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state, seg_predictions

    def preprocess_depth(self, depth, min_d, max_d):
        return habitat_depth_to_cm(
            depth,
            min_d,
            max_d,
            far_clip_value_m=100.0,
        )

    def pred_sem(self, rgb, depth=None, use_seg=True, pred_bbox=False):
        if pred_bbox:
            semantic_pred, self.rgb_vis, self.pred_box, seg_predictions = self.sem_pred.get_prediction(rgb)
            return self.pred_box, seg_predictions
        else:
            if use_seg:
                semantic_pred, self.rgb_vis, self.pred_box, seg_predictions = self.sem_pred.get_prediction(rgb)
                semantic_pred = semantic_pred.astype(np.float32)
                if depth is not None:
                    normalize_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    self.rgb_vis = cv2.cvtColor(normalize_depth, cv2.COLOR_GRAY2BGR)
            else:
                semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
                self.rgb_vis = rgb[:, :, ::-1]
            return semantic_pred, seg_predictions
        
    def get_goal_cat_id(self):
        if self.args.goal_type == 'ins-image':
            instance_whwh, seg_predictions = self.pred_sem(self.instance_imagegoal.astype(np.uint8), None, pred_bbox=True)
            ins_whwh = [instance_whwh[i] for i in range(len(instance_whwh)) \
                if (instance_whwh[i][2][3]-instance_whwh[i][2][1])>1/6*self.instance_imagegoal.shape[0] or \
                    (instance_whwh[i][2][2]-instance_whwh[i][2][0])>1/6*self.instance_imagegoal.shape[1]]
            if ins_whwh != []:
                ins_whwh = sorted(ins_whwh,  \
                    key=lambda s: ((s[2][0]+s[2][2]-self.instance_imagegoal.shape[1])/2)**2 \
                        +((s[2][1]+s[2][3]-self.instance_imagegoal.shape[0])/2)**2 \
                    )
                if ((ins_whwh[0][2][0]+ins_whwh[0][2][2]-self.instance_imagegoal.shape[1])/2)**2 \
                        +((ins_whwh[0][2][1]+ins_whwh[0][2][3]-self.instance_imagegoal.shape[0])/2)**2 < \
                            ((self.instance_imagegoal.shape[1] / 6)**2 )*2:
                    return int(ins_whwh[0][0])
            return None
        elif self.args.goal_type == 'text':
            intrinsic_text = format_text_goal_description(
                self.text_goal, intrinsic_only=True
            )
            full_text = format_text_goal_description(self.text_goal)
            parsed_id = infer_text_goal_category_id(intrinsic_text)
            if parsed_id is None:
                parsed_id = infer_text_goal_category_id(full_text)
            if parsed_id is not None:
                return int(parsed_id)
            for i in range(10):
                text_goal = full_text
                text_goal_id = self.llm(self.prompt_text2object.replace('{text}', text_goal))
                try:
                    text_goal_id = re.findall(r'\d+', text_goal_id)[0]
                    text_goal_id = int(text_goal_id)
                    if 0 <= text_goal_id < 6:
                        return text_goal_id
                except:
                    pass
            return -1

    def visualize(self, inputs):
        args = self.args

        color_palette = [
            1.0, 1.0, 1.0,
            0.6, 0.6, 0.6,
            0.95, 0.95, 0.95,
            0.96, 0.36, 0.26,
            0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
            0.9400000000000001, 0.7818, 0.66,
            0.8882000000000001, 0.9400000000000001, 0.66,
            0.66, 0.9400000000000001, 0.8518000000000001,
            0.7117999999999999, 0.66, 0.9400000000000001,
            0.9218, 0.66, 0.9400000000000001,
            0.9400000000000001, 0.66, 0.748199999999999]

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']

        goal = inputs['goal']
        sem_map = inputs['sem_map_pred']

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        # add a check with collision map
        map_pred[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 1

        sem_map += 5

        no_cat_mask = sem_map == 11
        # no_cat_mask = np.logical_or(no_cat_mask, 1 - no_cat_mask)
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1
        # vis_mask = self.visited[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        # <goal>
        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
        # </goal>

        if self.args.environment == 'habitat' and self.args.goal_type != 'goat':
            locs = np.array(self.envs._env.current_episode.goals[0].position[:2]) + np.array([18, 18])
            r, c = locs[1], locs[0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            if gx1 + 1 <= loc_c < gx2 - 1 and gy1 + 1 <= loc_r < gy2 - 1:
                sem_map[loc_r - gy1 - 1:loc_r - gy1 + 2, loc_c - gx1 - 1:loc_c - gx1 + 2] = [255, 0, 0]

        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        # sem_map_vis = insert_s_goal(self.s_goal, sem_map_vis, goal)
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        if self.args.environment == 'habitat':
            rgb_visualization = cv2.resize(self.rgb_vis, (360, 480), interpolation=cv2.INTER_NEAREST)

        vis_image = self.vis_image_background.copy()
        if self._is_instance_image_goal():
            instance_imagegoal = self.instance_imagegoal
            h, w = instance_imagegoal.shape[0], instance_imagegoal.shape[1]
            if h > w:
                instance_imagegoal = instance_imagegoal[h // 2 - w // 2:h // 2 + w // 2, :]
            elif w > h:
                instance_imagegoal = instance_imagegoal[:, w // 2 - h // 2:w // 2 + h // 2]
            instance_imagegoal = cv2.resize(instance_imagegoal, (215, 215), interpolation=cv2.INTER_NEAREST)
            instance_imagegoal = cv2.cvtColor(instance_imagegoal, cv2.COLOR_RGB2BGR)
            if self.args.environment == 'habitat':
                vis_image[50:265, 25:240] = instance_imagegoal
        elif self.args.goal_type == 'text' or (
            self.args.goal_type == 'goat' and self.text_goal is not None
        ):
            if isinstance(self.text_goal, dict) and 'intrinsic_attributes' in self.text_goal and 'extrinsic_attributes' in self.text_goal:
                text_goal = self.text_goal['intrinsic_attributes'] + ' ' + self.text_goal['extrinsic_attributes']
            else:
                text_goal = self.text_goal
            text_goal = line_list(text_goal)[:12]
            add_text_list(vis_image[50:265, 25:240], text_goal)
        elif self.args.goal_type in ('objnav', 'goat'):
            obj_lines = line_list(self.envs.goal_name)[:12]
            add_text_list(vis_image[50:265, 25:240], obj_lines)

        # Render Goal Graph text into the Goal Graph area (315:530, 25:240)
        goal_graph_text = inputs.get('goal_graph_text', '')
        if goal_graph_text:
            gg_lines = line_list(goal_graph_text)[:12]
            add_text_list(vis_image[315:530, 25:240], gg_lines)

        vis_image[50:530, 650:1130] = sem_map_vis
        if self.args.environment == 'habitat':
            vis_image[50:530, 265:625] = rgb_visualization
        if self.args.environment == 'habitat':
            cv2.rectangle(vis_image, (25, 50), (240, 265), (128, 128, 128), 1)
            cv2.rectangle(vis_image, (25, 315), (240, 530), (128, 128, 128), 1)
        cv2.rectangle(vis_image, (650, 50), (1130, 530), (128, 128, 128), 1)
        if self.args.environment == 'habitat':
            cv2.rectangle(vis_image, (265, 50), (625, 530), (128, 128, 128), 1)

        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        agent_arrow = get_contour_points(pos, origin=(885-200-10-25, 50))
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(vis_image, [agent_arrow], 0, color, -1)

        self.vis_image_list.append(vis_image)
        tmp_dir = 'outputs/tmp'
        os.makedirs(tmp_dir, exist_ok=True)
        height, width, layers = vis_image.shape
        if self.args.is_debugging:
            image_name = 'debug.jpg'
        else:
            image_name = 'v.jpg'
        cv2.imwrite(os.path.join(tmp_dir, image_name), cv2.resize(vis_image, (width // 2, height // 2)))
    
    def save_visualization(self, video_path):
        save_video(self.vis_image_list, video_path, fps=15, input_color_space="BGR")
        self.vis_image_list = []
