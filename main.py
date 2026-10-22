import os
import sys
import json
import logging
import time
import yaml
from collections import deque, defaultdict
from types import SimpleNamespace
import numpy as np
import torch
import argparse
from src.envs import construct_envs
from src.agent.unigoal.agent import UniGoal_Agent
from src.map.bev_mapping import BEV_Map
from src.graph.graph import Graph
from src.utils.cli_config import merge_config_with_cli
from src.utils.private_config import apply_model_environment, redact_config
import gzip

def get_config():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/config_habitat.yaml",
                        metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--goal_type", default="ins-image", choices=["ins-image", "text", "objnav", "goat"], type=str)
    parser.add_argument("--goat_split", default="val_seen",
                        choices=["val_seen", "val_seen_synonyms", "val_unseen"],
                        help="GOAT-Bench evaluation split")
    parser.add_argument("--episode_id", default=-1, type=int, help="episode id, 0~999")
    parser.add_argument("--goal", default="", type=str)
    parser.add_argument("--real_world", action="store_true")
    parser.add_argument("--disable_temporal", action="store_true",
                        help="Disable dynamic temporal scene graph (for ablation)")
    parser.add_argument("--disable_hypergraph", action="store_true",
                        help="Disable probabilistic hypergraph matching; fall back to overlap() (for ablation)")
    parser.add_argument("--disable_peek", action="store_true",
                        help="Disable semantic boundary pre-check / doorway peek (for ablation)")
    parser.add_argument("--disable_arrival_reject", action="store_true",
                        help="Disable arrival-verifier negative-evidence feedback to frontier scoring")
    parser.add_argument("--disable_frontier_belief", action="store_true",
                        help="Disable bounded frontier belief utility (for ablation)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume evaluation from the last saved checkpoint")
    parser.add_argument("--web_output_dir", default="", type=str,
                        help="If set, export graph snapshots + frames to this dir for web UI")

    parsed_args = parser.parse_args()

    with open(parsed_args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    args = merge_config_with_cli(
        parser,
        parsed_args,
        config,
        argv=sys.argv[1:],
    )
        
    args = SimpleNamespace(**apply_model_environment(args))

    args.is_debugging = sys.gettrace() is not None
    if args.is_debugging:
        args.experiment_id = "debug"
    if (
        args.goal_type == 'goat'
        and getattr(args, 'episode_id', -1) >= 0
        and not getattr(args, 'resume', False)
    ):
        args.start_episode = int(args.episode_id)
        args.num_eval_episodes = 1
        args.experiment_id = '{}_episode_{:0>6}'.format(
            args.experiment_id, int(args.episode_id)
        )
    else:
        args.start_episode = 0
    
    args.log_dir = os.path.join(args.dump_location, args.experiment_id, 'log')
    args.visualization_dir = os.path.join(args.dump_location, args.experiment_id, 'visualization')

    args.map_size = args.map_size_cm // args.map_resolution
    args.global_width, args.global_height = args.map_size, args.map_size
    args.local_width = int(args.global_width / args.global_downscaling)
    args.local_height = int(args.global_height / args.global_downscaling)

    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    args.num_scenes = args.num_processes
    args.num_episodes = int(args.num_eval_episodes)

    return args


def _set_goat_goal(graph, infos):
    """Set graph goal based on the current GOAT subtask info."""
    from src.envs.goat_episode_parser import (
        GOAL_TYPE_OBJECT, GOAL_TYPE_DESCRIPTION, GOAL_TYPE_IMAGE
    )
    graph.start_new_goal()
    goal_type = infos.get('goal_type', GOAL_TYPE_OBJECT)
    graph.current_goat_goal_type = goal_type
    graph.set_obj_goal(infos['goal_name'])
    if goal_type == GOAL_TYPE_IMAGE and 'instance_imagegoal' in infos:
        graph.set_image_goal(infos['instance_imagegoal'])
    elif goal_type == GOAL_TYPE_DESCRIPTION and 'text_goal' in infos:
        graph.set_text_goal(infos['text_goal'])
    else:
        graph.goalgraph = {'nodes': [{'id': infos['goal_name']}], 'edges': []}
        graph.goalgraph_decomposed = {'subgraph_1': graph.goalgraph}
        graph.text_goal = infos['goal_name']


def _global_goal_to_local_waypoint(goal, boundary, local_width, local_height):
    goal_arr = np.asarray(goal, dtype=float).reshape(-1)
    if goal_arr.size < 2 or not np.all(np.isfinite(goal_arr[:2])):
        return None
    gx1, _, gy1, _ = boundary
    local_goal = [
        int(round(float(goal_arr[0]) - gx1)),
        int(round(float(goal_arr[1]) - gy1)),
    ]
    in_window = 0 <= local_goal[0] < local_width and 0 <= local_goal[1] < local_height
    if in_window:
        return local_goal
    return [
        int(np.clip(local_goal[0], 0, local_width - 1)),
        int(np.clip(local_goal[1], 0, local_height - 1)),
    ]


def _project_local_goal_to_free(goal, obstacle_map, search_radius=25):
    if goal is None:
        return None
    obstacle = np.rint(obstacle_map).astype(bool)
    h, w = obstacle.shape
    row = int(np.clip(goal[0], 0, h - 1))
    col = int(np.clip(goal[1], 0, w - 1))
    if not obstacle[row, col]:
        return [row, col]

    best = None
    best_dist = float('inf')
    for radius in range(1, search_radius + 1):
        r1, r2 = max(0, row - radius), min(h, row + radius + 1)
        c1, c2 = max(0, col - radius), min(w, col + radius + 1)
        rr, cc = np.where(~obstacle[r1:r2, c1:c2])
        if len(rr) == 0:
            continue
        rr = rr + r1
        cc = cc + c1
        dist = np.hypot(rr - row, cc - col)
        idx = int(np.argmin(dist))
        if float(dist[idx]) < best_dist:
            best_dist = float(dist[idx])
            best = [int(rr[idx]), int(cc[idx])]
            break
    return best


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _append_frontier_trace(
    trace_path,
    step,
    infos,
    graph,
    selected_goal=None,
    local_goal=None,
    reason='global_replan',
    semantic_direct_goal=None,
    rejected_count=0,
    agent=None,
):
    target_evidence = {}
    arrival_verdict = {}
    if agent is not None:
        target_evidence = getattr(agent, 'current_target_evidence', {}) or {}
        arrival_verdict = getattr(agent, 'last_arrival_verdict', {}) or {}
    record = {
        'step': int(step),
        'episode_no': infos.get('episode_no'),
        'subtask_idx': infos.get('subtask_idx'),
        'goal_name': infos.get('goal_name'),
        'goal_type': infos.get('goal_type'),
        'reason': reason,
        'selected_global_goal': selected_goal,
        'selected_local_goal': local_goal,
        'semantic_direct_goal': semantic_direct_goal,
        'rejected_count': int(rejected_count),
        'target_evidence': target_evidence,
        'arrival_verdict': arrival_verdict,
        'frontier_debug': getattr(graph, 'frontier_debug', {}) or {},
    }
    try:
        with open(trace_path, 'a', encoding='utf-8') as trace_file:
            trace_file.write(json.dumps(_json_safe(record)) + '\n')
    except Exception as exc:
        logging.warning('Failed to write frontier trace: %s', exc)


def main():
    args = get_config()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.visualization_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.log_dir, 'eval.log'),
        level=logging.INFO)
    logging.info('Configuration: %s', redact_config(vars(args)))

    eval_metrics_id = 0

    episode_success = deque(maxlen=args.num_episodes)
    episode_spl = deque(maxlen=args.num_episodes)

    # GOAT fine-grained metrics (chain_sr, per-type SR/SPL)
    is_goat = args.goal_type == 'goat'
    goat_chain_sr = deque(maxlen=args.num_episodes)
    goat_episode_sr = deque(maxlen=args.num_episodes)
    goat_type_results = {"object": [], "description": [], "image": []}
    goat_subtask_results = []

    finished = False
    wait_env = False

    # ---- checkpoint / resume ----
    checkpoint_path = os.path.join(args.log_dir, 'checkpoint.json')
    goat_trace_path = os.path.join(args.log_dir, 'goat_trace.jsonl')
    frontier_trace_path = os.path.join(args.log_dir, 'frontier_trace.jsonl')
    checkpoint_results = []
    args.start_episode = getattr(args, 'start_episode', 0)
    if is_goat and not args.resume and os.path.exists(goat_trace_path):
        os.remove(goat_trace_path)
    if not args.resume and os.path.exists(frontier_trace_path):
        os.remove(frontier_trace_path)
    if args.resume and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as _f:
                _ckpt = json.load(_f)
            checkpoint_results = _ckpt.get('results', [])
            args.start_episode = len(checkpoint_results)
            if args.start_episode >= args.num_episodes:
                print('[Resume] All episodes already completed.')
                _s = [r['success'] for r in checkpoint_results]
                _p = [r['spl'] for r in checkpoint_results]
                print('Average SR/SPL: {:.5f}/{:.5f}'.format(np.mean(_s), np.mean(_p)))
                return
            for _r in checkpoint_results:
                episode_success.append(_r['success'])
                episode_spl.append(_r['spl'])
                goat_subtask_results.extend(_r.get('goat_subtask_results', []))
                if is_goat and 'goat_chain_sr' in _r:
                    goat_chain_sr.append(_r['goat_chain_sr'])
                    if 'goat_episode_sr' in _r:
                        goat_episode_sr.append(_r['goat_episode_sr'])
                    for _type, _sr in _r.get('goat_sr_by_type', {}).items():
                        if _type in goat_type_results:
                            goat_type_results[_type].append(_sr)
            eval_metrics_id = args.start_episode
            print(f'[Resume] Loaded checkpoint: {args.start_episode}/{args.num_episodes} episodes completed, resuming from episode {args.start_episode}.')
            logging.info(f'[Resume] Resuming from episode {args.start_episode}')
        except Exception as _e:
            print(f'[Resume] Failed to load checkpoint ({_e}), starting from scratch.')
            checkpoint_results = []
            args.start_episode = 0
    # ---- end checkpoint / resume ----

    if args.goal_type == 'text':
        with gzip.open(args.text_goal_dataset, 'rt') as f:
            text_goal_dataset = json.load(f)

    BEV_map = BEV_Map(args)
    graph = Graph(args)
    envs = construct_envs(args)
    if is_goat:
        graph.set_open_vocab_categories(envs.goat_categories)
    agent = UniGoal_Agent(args, envs)

    BEV_map.init_map_and_pose()
    obs, rgbd, infos = agent.reset()

    BEV_map.mapping(rgbd, infos)

    global_goals = [args.local_width // 2, args.local_height // 2]

    goal_maps = np.zeros((args.local_width, args.local_height))

    goal_maps[global_goals[0], global_goals[1]] = 1

    agent_input = {}
    agent_input['map_pred'] = BEV_map.local_map[0, 0, :, :].cpu().numpy()
    agent_input['exp_pred'] = BEV_map.local_map[0, 1, :, :].cpu().numpy()
    agent_input['pose_pred'] = BEV_map.planner_pose_inputs[0]
    agent_input['goal'] = goal_maps
    agent_input['exp_goal'] = goal_maps * 1
    agent_input['new_goal'] = 1
    agent_input['found_goal'] = 0
    agent_input['wait'] = wait_env or finished
    agent_input['sem_map'] = BEV_map.local_map[0, 4:11, :, :
                                        ].cpu().numpy()
    if args.visualize:
        BEV_map.local_map[0, 10, :, :] = 1e-5
        agent_input['sem_map_pred'] = BEV_map.local_map[0, 4:11, :, :
                                            ].argmax(0).cpu().numpy()

    obs, rgbd, done, infos = agent.step(agent_input)

    graph.reset()
    graph.set_obj_goal(infos['goal_name'])
    if args.goal_type == 'ins-image':
        graph.set_image_goal(infos['instance_imagegoal'])
    elif args.goal_type == 'text':
        graph.set_text_goal(infos['text_goal'])
    elif args.goal_type == 'objnav':
        graph.goalgraph = {'nodes': [{'id': infos['goal_name']}], 'edges': []}
        graph.goalgraph_decomposed = {'subgraph_1': graph.goalgraph}
        graph.text_goal = infos['goal_name']
    elif args.goal_type == 'goat':
        _set_goat_goal(graph, infos)

    step = 0
    prev_subtask_idx = 0  # tracks GOAT subtask changes within an episode

    while True:
        if finished == True:
            break

        local_step = step % args.num_local_steps

        # ---- GOAT: detect within-episode subtask switch ----
        if is_goat and not done:
            cur_subtask_idx = infos.get('subtask_idx', 0)
            if cur_subtask_idx != prev_subtask_idx:
                prev_subtask_idx = cur_subtask_idx
                agent.update_goat_goal(infos)
                _set_goat_goal(graph, infos)
                print(f"[GOAT] subtask switch → idx={cur_subtask_idx}  "
                      f"type={infos.get('goal_type')}  cat={infos.get('goal_name')}")
        # ----------------------------------------------------

        if done:
            episode_result_info = infos.get('_episode_result_info', infos)
            spl = episode_result_info.get('spl') or 0.0
            success = episode_result_info.get('success') or 0.0
            eval_metrics_id += 1
            episode_success.append(success)
            episode_spl.append(spl)

            ckpt_entry = {'episode_no': episode_result_info['episode_no'],
                          'success': float(success),
                          'spl': float(spl)}
            for _key in ('distance_to_goal', 'soft_spl', 'path_length'):
                _value = episode_result_info.get(_key)
                if _value is not None:
                    try:
                        ckpt_entry[_key] = float(_value)
                    except (TypeError, ValueError):
                        pass
            for _key in ('num_steps', 'max_episode_steps', 'last_action'):
                _value = episode_result_info.get(_key)
                if _value is not None:
                    try:
                        ckpt_entry[_key] = int(_value)
                    except (TypeError, ValueError):
                        pass
            if 'stopped' in episode_result_info:
                ckpt_entry['stopped'] = bool(episode_result_info.get('stopped'))
            for _key in ('goal_name', 'goal_type', 'subtask_idx'):
                if _key in episode_result_info:
                    ckpt_entry[_key] = episode_result_info.get(_key)
            if hasattr(args, 'success_dist'):
                ckpt_entry['success_dist'] = float(args.success_dist)
            if is_goat and 'goat_metrics' in episode_result_info:
                gm = episode_result_info['goat_metrics']
                ckpt_entry['goat_chain_sr'] = gm['chain_sr']
                ckpt_entry['goat_episode_sr'] = gm['episode_sr']
                ckpt_entry['goat_sr_by_type'] = gm['sr_by_type']
                ckpt_entry['goat_spl_by_type'] = gm['spl_by_type']
                ckpt_entry['goat_subtask_results'] = gm['subtask_results']
                goat_subtask_results.extend(gm['subtask_results'])
                goat_chain_sr.append(gm['chain_sr'])
                goat_episode_sr.append(gm['episode_sr'])
                for gtype, sr in gm['sr_by_type'].items():
                    if gtype in goat_type_results:
                        goat_type_results[gtype].append(sr)
                with open(goat_trace_path, 'a') as _tf:
                    for _item in gm['subtask_results']:
                        _trace_item = {
                            'episode_no': episode_result_info['episode_no'],
                            **_item,
                        }
                        _tf.write(json.dumps(_trace_item) + '\n')

            checkpoint_results.append(ckpt_entry)
            _tmp = checkpoint_path + '.tmp'
            with open(_tmp, 'w') as _cf:
                json.dump({'results': checkpoint_results}, _cf)
            os.replace(_tmp, checkpoint_path)
            if len(episode_success) == args.num_episodes:
                finished = True
            if args.visualize:
                video_path = os.path.join(args.visualization_dir, 'videos', 'eps_{:0>6}.mp4'.format(episode_result_info['episode_no']))
                agent.save_visualization(video_path)
            if finished:
                break
            wait_env = True
            BEV_map.update_intrinsic_rew()
            BEV_map.init_map_and_pose_for_env()

            # For GOAT: graph resets only at episode boundary (already handled by done=True)
            # For single-goal tasks: reset as before
            graph.reset()
            graph.set_obj_goal(infos['goal_name'])
            if args.goal_type == 'ins-image':
                graph.set_image_goal(infos['instance_imagegoal'])
            elif args.goal_type == 'text':
                graph.set_text_goal(infos['text_goal'])
            elif args.goal_type == 'objnav':
                graph.goalgraph = {'nodes': [{'id': infos['goal_name']}], 'edges': []}
                graph.goalgraph_decomposed = {'subgraph_1': graph.goalgraph}
                graph.text_goal = infos['goal_name']
            elif args.goal_type == 'goat':
                _set_goat_goal(graph, infos)
            prev_subtask_idx = 0

        BEV_map.mapping(rgbd, infos)

        navigate_steps = step
        graph.set_navigate_steps(navigate_steps)
        semantic_direct_goal = None
        scenegraph_updated = False
        if not agent_input['wait'] and navigate_steps % 2 == 0:
            graph.set_observations(obs)
            graph.update_scenegraph()
            scenegraph_updated = True

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        did_global_replan = (
            local_step == args.num_local_steps - 1
            or np.linalg.norm(
                np.array([BEV_map.local_row, BEV_map.local_col])
                - np.array(global_goals)
            ) < 10
        )
        if did_global_replan:
            if wait_env == True:
                wait_env = False
            else:
                BEV_map.update_intrinsic_rew()

            BEV_map.move_local_map()

            graph.set_full_map(BEV_map.full_map)
            graph.set_full_pose(BEV_map.full_pose)
            if hasattr(graph, 'set_rejected_goal_regions'):
                graph.set_rejected_goal_regions(getattr(agent, 'rejected_goals', []))
            goal = graph.explore()
            if is_goat and getattr(graph, '_semantic_goal_selected_direct', False):
                semantic_direct_goal = np.asarray(goal, dtype=float).tolist()
            frontier_locations_16 = getattr(graph, 'frontier_locations_16', None)
            if frontier_locations_16 is not None and len(frontier_locations_16) > 0:
                graph.frontier_locations_16_local = frontier_locations_16.copy()
                graph.frontier_locations_16_local[:, 0] -= BEV_map.local_map_boundary[0, 0]
                graph.frontier_locations_16_local[:, 1] -= BEV_map.local_map_boundary[0, 2]
            else:
                graph.frontier_locations_16_local = np.empty((0, 2), dtype=np.float32)
            if isinstance(goal, list) or isinstance(goal, np.ndarray):
                goal = list(goal)
                local_waypoint = _global_goal_to_local_waypoint(
                    goal, BEV_map.local_map_boundary[0],
                    args.local_width, args.local_height
                )
                if local_waypoint is not None:
                    projected_waypoint = _project_local_goal_to_free(
                        local_waypoint,
                        BEV_map.local_map[0, 0, :, :].cpu().numpy(),
                    )
                    if projected_waypoint is not None:
                        global_goals = projected_waypoint
            _append_frontier_trace(
                frontier_trace_path,
                step,
                infos,
                graph,
                selected_goal=goal,
                local_goal=global_goals,
                reason='global_replan',
                semantic_direct_goal=semantic_direct_goal,
                rejected_count=len(getattr(agent, 'rejected_goals', [])),
                agent=agent,
            )
        elif is_goat and scenegraph_updated:
            # Do not wait for the next frontier cycle if the open-vocabulary
            # scene graph has already seen the current GOAT target in this
            # episode.  This is a lightweight recall path; full frontier
            # scoring still happens in the regular global replan branch.
            BEV_map.update_intrinsic_rew()
            graph.set_full_map(BEV_map.full_map)
            graph.set_full_pose(BEV_map.full_pose)
            if hasattr(graph, 'set_rejected_goal_regions'):
                graph.set_rejected_goal_regions(getattr(agent, 'rejected_goals', []))
            remembered_goal = graph.recall_current_goal_from_memory()
            if remembered_goal is not None:
                semantic_direct_goal = np.asarray(remembered_goal, dtype=float).tolist()
                _append_frontier_trace(
                    frontier_trace_path,
                    step,
                    infos,
                    graph,
                    selected_goal=semantic_direct_goal,
                    local_goal=global_goals,
                    reason='goat_memory_recall',
                    semantic_direct_goal=semantic_direct_goal,
                    rejected_count=len(getattr(agent, 'rejected_goals', [])),
                    agent=agent,
                )


        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        found_goal = False
        goal_maps = np.zeros((args.local_width, args.local_height))

        global_goals = [
            int(np.clip(global_goals[0], 0, args.local_width - 1)),
            int(np.clip(global_goals[1], 0, args.local_height - 1)),
        ]
        goal_maps[global_goals[0], global_goals[1]] = 1

        exp_goal_maps = goal_maps.copy()

        agent_input = {}
        agent_input['map_pred'] = BEV_map.local_map[0, 0, :, :].cpu().numpy()
        agent_input['exp_pred'] = BEV_map.local_map[0, 1, :, :].cpu().numpy()
        agent_input['pose_pred'] = BEV_map.planner_pose_inputs[0]
        agent_input['goal'] = goal_maps
        agent_input['exp_goal'] = exp_goal_maps
        agent_input['new_goal'] = local_step == args.num_local_steps - 1
        agent_input['found_goal'] = found_goal
        agent_input['semantic_direct_goal'] = semantic_direct_goal
        agent_input['wait'] = wait_env or finished
        agent_input['sem_map'] = BEV_map.local_map[0, 4:11, :, :
                                        ].cpu().numpy()
        
        if hasattr(graph, 'goalgraph') and graph.goalgraph:
            agent_input['goal_graph_text'] = graph.goalgraphdecomposer.graph_to_text(graph.goalgraph)
        else:
            agent_input['goal_graph_text'] = ""

        # 传递场景图节点（caption + 地图坐标）和目标图给到达验证器
        agent_input['scene_nodes'] = [
            {'caption': n.caption, 'center': n.center}
            for n in graph.nodes if n.center is not None
        ]
        agent_input['goalgraph'] = graph.goalgraph if hasattr(graph, 'goalgraph') else None
        agent_input['text_goal'] = graph.text_goal if hasattr(graph, 'text_goal') else None

        if args.visualize:
            BEV_map.local_map[0, 10, :, :] = 1e-5
            agent_input['sem_map_pred'] = BEV_map.local_map[0, 4:11, :,
                                                :].argmax(0).cpu().numpy()

        obs, rgbd, done, infos = agent.step(agent_input)

        # ------------------------------------------------------------------

        # ── Web UI export: frames + graph snapshots ──────────────────────
        _web_dir = getattr(args, 'web_output_dir', '')
        _snap_interval = getattr(args, 'snapshot_interval', 10)
        if _web_dir and step % _snap_interval == 0 and not agent_input['wait']:
            try:
                import cv2 as _cv2
                # ---- Save current visualization frame ----
                _frames_dir = os.path.join(_web_dir, 'frames')
                os.makedirs(_frames_dir, exist_ok=True)
                _vis_jpg = os.path.join('outputs', 'tmp', 'v.jpg')
                if os.path.exists(_vis_jpg):
                    import shutil
                    shutil.copy2(_vis_jpg, os.path.join(_frames_dir, 'step_{:05d}.jpg'.format(step)))

                # ---- Export graph snapshot ----
                _graphs_dir = os.path.join(_web_dir, 'graphs')
                os.makedirs(_graphs_dir, exist_ok=True)
                _pose = BEV_map.planner_pose_inputs[0]
                _snapshot = {
                    'step': step,
                    'agent_pose': {
                        'x': float(_pose[0]),
                        'y': float(_pose[1]),
                        'heading': float(_pose[2]),
                    },
                    'nodes': [],
                    'edges': [],
                }
                # Export nodes
                _uid_map = {}
                for _idx, _n in enumerate(graph.nodes):
                    _uid = '{}_{}'.format(_n.caption, id(_n))
                    _uid_map[id(_n)] = _uid
                    _snapshot['nodes'].append({
                        'uid': _uid,
                        'caption': _n.caption,
                        'center': list(_n.center) if _n.center is not None else [0, 0],
                        'confidence': float(_n.confidence),
                        'n_obs': int(_n.n_obs),
                        'c_trend': float(_n.c_trend),
                        'is_in_fov': bool(getattr(_n, 'is_in_fov', False)),
                    })
                # Export edges
                _seen = set()
                for _n in graph.nodes:
                    for _e in _n.edges:
                        _ek = tuple(sorted([id(_e.node1), id(_e.node2)]))
                        if _ek not in _seen and _e.relation is not None:
                            _seen.add(_ek)
                            _snapshot['edges'].append({
                                'src': _uid_map.get(id(_e.node1), ''),
                                'dst': _uid_map.get(id(_e.node2), ''),
                                'relation': str(_e.relation) if _e.relation else '',
                                'weight': float(_e.weight),
                                'n_simul': int(getattr(_e, 'n_simul_observed', 0)),
                            })
                # Hyperedges from prob_map
                if hasattr(graph, 'prob_map') and graph.prob_map:
                    _snapshot['prob_map'] = {k: float(v) for k, v in graph.prob_map.items()}
                if hasattr(graph, 'frontier_debug') and graph.frontier_debug:
                    _snapshot['frontier_debug'] = graph.frontier_debug
                if is_goat and hasattr(graph, 'goal_memory'):
                    _snapshot['goal_memory'] = {
                        _key: [
                            {
                                'caption': _entry.get('caption', ''),
                                'center': list(_entry.get('center', [0, 0])),
                                'confidence': float(_entry.get('confidence', 0.0)),
                                'last_seen': int(_entry.get('last_seen', -1)),
                                'n_obs': int(_entry.get('n_obs', 0)),
                            }
                            for _entry in _entries
                        ]
                        for _key, _entries in graph.goal_memory.items()
                    }
                # Frontiers
                _frontiers = getattr(graph, 'frontier_locations_16_local', None)
                if _frontiers is not None:
                    _snapshot['frontiers'] = [
                        {'pos': [float(f[0]), float(f[1])], 'score': 0.0}
                        for f in _frontiers
                    ]
                with open(os.path.join(_graphs_dir, 'step_{:05d}.json'.format(step)), 'w') as _sf:
                    json.dump(_snapshot, _sf)
            except Exception as _web_err:
                logging.warning('Web export error at step {}: {}'.format(step, _web_err))
        # ─────────────────────────────────────────────────────────────────

        # ------------------------------------------------------------------
        # log
        if step % args.log_interval == 0:
            log = " ".join([
                "num timesteps {},".format(step),
                "episode_id {}".format(infos['episode_no']),
            ])

            total_success = []
            total_spl = []
            for acc in episode_success:
                total_success.append(acc)
            for spl in episode_spl:
                total_spl.append(spl)

            if len(total_spl) > 0:
                if is_goat:
                    log += " GOAT episode_macro_subtask_SR/SPL:"
                else:
                    log += " Average SR/SPL:"
                log += " {:.5f}/{:.5f},".format(
                    np.mean(total_success),
                    np.mean(total_spl))
                if is_goat and goat_subtask_results:
                    log += " GOAT subtask_micro_SR/SPL: {:.5f}/{:.5f},".format(
                        np.mean([item['success'] for item in goat_subtask_results]),
                        np.mean([item['spl'] for item in goat_subtask_results]),
                    )
                    if goat_episode_sr:
                        log += " GOAT all_subtasks_episode_SR: {:.5f},".format(
                            np.mean(list(goat_episode_sr))
                        )

            print(log)
            logging.info(log)
        # ------------------------------------------------------------------
        step += 1

    total_success = []
    total_spl = []
    for acc in episode_success:
        total_success.append(acc)
    for spl in episode_spl:
        total_spl.append(spl)

    if len(total_spl) > 0:
        if is_goat:
            log = "GOAT episode_macro_subtask_SR/SPL:"
        else:
            log = "Average SR/SPL:"
        log += " {:.5f}/{:.5f}".format(
            np.mean(total_success),
            np.mean(total_spl))
        if is_goat and len(goat_chain_sr) > 0:
            log += "  GOAT chain_SR: {:.5f}".format(np.mean(list(goat_chain_sr)))
            if goat_episode_sr:
                log += "  GOAT all_subtasks_episode_SR: {:.5f}".format(
                    np.mean(list(goat_episode_sr))
                )
            if goat_subtask_results:
                log += "  GOAT subtask_SR/SPL: {:.5f}/{:.5f}".format(
                    np.mean([item['success'] for item in goat_subtask_results]),
                    np.mean([item['spl'] for item in goat_subtask_results]),
                )
                for gtype in ("object", "description", "image"):
                    typed = [item for item in goat_subtask_results if item['goal_type'] == gtype]
                    if typed:
                        log += "  {}_SR/SPL: {:.5f}/{:.5f}".format(
                            gtype,
                            np.mean([item['success'] for item in typed]),
                            np.mean([item['spl'] for item in typed]),
                        )

    print(log)
    logging.info(log)

    total = {'succ': total_success, 'spl': total_spl}
    if is_goat:
        total['goat_chain_sr'] = list(goat_chain_sr)
        total['goat_episode_sr'] = list(goat_episode_sr)
        total['goat_type_sr'] = {k: v for k, v in goat_type_results.items() if v}
        total['goat_subtask_results'] = goat_subtask_results
        if goat_subtask_results:
            total['goat_subtask_sr'] = float(np.mean(
                [item['success'] for item in goat_subtask_results]
            ))
            total['goat_subtask_spl'] = float(np.mean(
                [item['spl'] for item in goat_subtask_results]
            ))

    with open('{}/total.json'.format(
            args.log_dir), 'w') as f:
        json.dump(total, f)


if __name__ == "__main__":
    main()
