"""Load and resolve the released GOAT-Bench episode format.

Each per-scene file contains two top-level dictionaries:

``episodes``
    Compact task references.  Object and description tasks are encoded as
    ``[category, type, instance_id]``.  Image tasks add a fourth element with
    the selected image-goal index.

``goals``
    The instance database containing language descriptions, image rendering
    parameters and valid navigation viewpoints.  Compact task references are
    unusable until they are resolved against this database.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


GOAL_TYPE_OBJECT = "object"
GOAL_TYPE_DESCRIPTION = "description"
GOAL_TYPE_IMAGE = "image"


@dataclass
class GoatSubtask:
    task_id: int
    goal_type: str
    object_category: str
    object_id: Optional[str]
    goal_instances: List[dict]
    view_points: List[dict]
    lang_desc: Optional[str] = None
    image_goal: Optional[dict] = None
    image_goal_index: Optional[int] = None


@dataclass
class GoatEpisode:
    episode_id: str
    scene_id: str
    start_position: List[float]
    start_rotation: List[float]
    scene_dataset_config: Optional[str] = None
    subtasks: List[GoatSubtask] = field(default_factory=list)


class GoatEpisodeParser:
    """Resolve GOAT split metadata and per-scene content files."""

    def __init__(self, dataset_path: str, goat_image_dir: str = ""):
        # goat_image_dir is retained for CLI compatibility.  Released GOAT
        # image goals are camera parameters, not image file paths.
        self.dataset_path = dataset_path
        self.goat_image_dir = goat_image_dir
        self.episodes: List[GoatEpisode] = []
        self.categories: List[str] = []
        self.content_files: List[str] = []
        self._load()

    @staticmethod
    def _read_json(path: str) -> dict:
        open_fn = gzip.open if path.endswith(".gz") else open
        with open_fn(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _normalise_goal_type(value: str) -> str:
        value = (value or GOAL_TYPE_OBJECT).lower().strip()
        aliases = {
            "objectnav": GOAL_TYPE_OBJECT,
            "object_nav": GOAL_TYPE_OBJECT,
            "object_goal": GOAL_TYPE_OBJECT,
            "objectgoal": GOAL_TYPE_OBJECT,
            "language": GOAL_TYPE_DESCRIPTION,
            "languagenav": GOAL_TYPE_DESCRIPTION,
            "language_nav": GOAL_TYPE_DESCRIPTION,
            "text": GOAL_TYPE_DESCRIPTION,
            "text_goal": GOAL_TYPE_DESCRIPTION,
            "lang_goal": GOAL_TYPE_DESCRIPTION,
            "instance_image": GOAL_TYPE_IMAGE,
            "imagenav": GOAL_TYPE_IMAGE,
            "image_nav": GOAL_TYPE_IMAGE,
            "image_goal": GOAL_TYPE_IMAGE,
            "instanceimagenav": GOAL_TYPE_IMAGE,
            "instance_imagenav": GOAL_TYPE_IMAGE,
        }
        value = aliases.get(value, value)
        if value not in {GOAL_TYPE_OBJECT, GOAL_TYPE_DESCRIPTION, GOAL_TYPE_IMAGE}:
            raise ValueError(f"Unsupported GOAT goal type: {value}")
        return value

    def _load(self) -> None:
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f"GOAT split file not found: {self.dataset_path}")

        split_data = self._read_json(self.dataset_path)
        records: List[Tuple[dict, Dict[str, Sequence[dict]]]] = []

        # Some generated variants keep episodes in the split file.  The public
        # release keeps only metadata there and stores 10 episodes per scene in
        # content/*.json.gz.
        if split_data.get("episodes"):
            goals = split_data.get("goals", {})
            records.extend((episode, goals) for episode in split_data["episodes"])
        else:
            split_dir = os.path.dirname(self.dataset_path)
            content_dir = os.path.join(split_dir, "content")
            if not os.path.isdir(content_dir):
                raise FileNotFoundError(
                    f"GOAT split has no inline episodes and no content directory: {content_dir}"
                )
            self.content_files = sorted(
                os.path.join(content_dir, name)
                for name in os.listdir(content_dir)
                if name.endswith((".json", ".json.gz"))
            )
            for content_file in self.content_files:
                scene_data = self._read_json(content_file)
                goals = scene_data.get("goals")
                if not isinstance(goals, dict):
                    raise ValueError(f"Missing top-level goals dictionary: {content_file}")
                records.extend((episode, goals) for episode in scene_data.get("episodes", []))

        categories = set()
        for episode, goals in records:
            parsed = self._parse_episode(episode, goals)
            self.episodes.append(parsed)
            categories.update(task.object_category for task in parsed.subtasks)
        self.categories = sorted(categories)

        print(
            f"[GoatEpisodeParser] Loaded {len(self.episodes)} episodes, "
            f"{sum(len(ep.subtasks) for ep in self.episodes)} subtasks and "
            f"{len(self.categories)} categories from {self.dataset_path}"
        )

    @staticmethod
    def _deduplicate_instances(instances: Iterable[dict]) -> List[dict]:
        result: List[dict] = []
        seen = set()
        for goal in instances:
            key = goal.get("object_id", id(goal))
            if key not in seen:
                seen.add(key)
                result.append(goal)
        return result

    def _category_instances(
        self, category: str, all_goals: Dict[str, Sequence[dict]]
    ) -> List[dict]:
        groups = [
            list(group)
            for group in all_goals.values()
            if group and group[0].get("object_category") == category
        ]
        if not groups:
            raise ValueError(f"No GOAT goal instances found for category '{category}'")

        instances = list(groups[0])
        # GOAT object goals include compatible child annotation categories.
        children = set()
        for goal in groups[0]:
            children.update(goal.get("children_object_categories") or [])
        for child in children:
            for group in all_goals.values():
                if group and group[0].get("object_category") == child:
                    instances.extend(group)
        return self._deduplicate_instances(instances)

    def _parse_episode(
        self, episode: dict, all_goals: Dict[str, Sequence[dict]]
    ) -> GoatEpisode:
        subtasks: List[GoatSubtask] = []
        raw_tasks = episode.get("tasks", episode.get("goal_sequence", []))

        for task_id, raw_task in enumerate(raw_tasks):
            if isinstance(raw_task, list):
                if len(raw_task) < 3:
                    raise ValueError(f"Malformed GOAT task: {raw_task}")
                category = str(raw_task[0])
                goal_type = self._normalise_goal_type(raw_task[1])
                object_id = None if raw_task[2] is None else str(raw_task[2])
                image_goal_index = raw_task[3] if len(raw_task) > 3 else None
            elif isinstance(raw_task, dict):
                category = str(
                    raw_task.get("object_category")
                    or raw_task.get("category")
                    or raw_task.get("goal_category")
                )
                goal_type = self._normalise_goal_type(
                    raw_task.get("goal_type")
                    or raw_task.get("task_type")
                    or raw_task.get("type")
                )
                object_id = raw_task.get("object_id")
                object_id = None if object_id is None else str(object_id)
                image_goal_index = raw_task.get("image_goal_index")
            else:
                raise ValueError(f"Unsupported GOAT task record: {raw_task!r}")

            category_instances = self._category_instances(category, all_goals)
            if goal_type == GOAL_TYPE_OBJECT:
                selected_instances = category_instances
            else:
                selected_instances = [
                    goal for goal in category_instances
                    if str(goal.get("object_id")) == object_id
                ]
                if len(selected_instances) != 1:
                    raise ValueError(
                        f"Expected one instance for {category}/{goal_type}/{object_id}, "
                        f"found {len(selected_instances)}"
                    )

            view_points = [
                viewpoint
                for goal in selected_instances
                for viewpoint in (goal.get("view_points") or [])
            ]
            if not view_points:
                raise ValueError(
                    f"GOAT task has no valid viewpoints: {category}/{goal_type}/{object_id}"
                )

            lang_desc = None
            image_goal = None
            if goal_type == GOAL_TYPE_DESCRIPTION:
                lang_desc = selected_instances[0].get("lang_desc")
                if not lang_desc:
                    raise ValueError(f"Missing language description for {object_id}")
            elif goal_type == GOAL_TYPE_IMAGE:
                image_goals = selected_instances[0].get("image_goals") or []
                if not isinstance(image_goal_index, int) or not (
                    0 <= image_goal_index < len(image_goals)
                ):
                    raise ValueError(
                        f"Invalid image goal index {image_goal_index} for {object_id}; "
                        f"available={len(image_goals)}"
                    )
                image_goal = image_goals[image_goal_index]

            subtasks.append(
                GoatSubtask(
                    task_id=task_id,
                    goal_type=goal_type,
                    object_category=category,
                    object_id=object_id,
                    goal_instances=selected_instances,
                    view_points=view_points,
                    lang_desc=lang_desc,
                    image_goal=image_goal,
                    image_goal_index=image_goal_index,
                )
            )

        return GoatEpisode(
            episode_id=str(episode.get("episode_id", "")),
            scene_id=episode.get("scene_id", ""),
            start_position=list(episode.get("start_position", [0.0, 0.0, 0.0])),
            start_rotation=list(episode.get("start_rotation", [0.0, 0.0, 0.0, 1.0])),
            scene_dataset_config=episode.get("scene_dataset_config"),
            subtasks=subtasks,
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> GoatEpisode:
        return self.episodes[index]
