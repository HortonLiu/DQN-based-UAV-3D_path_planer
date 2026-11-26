"""Deterministic planar scenes for the manuscript's baseline protocol.

The default XML contains the same 26 ordered radius/height pairs as the
available DDPG and TD3 archives. Its largest radius is 45.807 m, compared with
45.506 m reported in the manuscript. The original per-episode scenes and the
exact historical dimensions have not been recovered. This generator provides
one shared configuration for new runs.
"""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


DEFAULT_DIMENSIONS_XML = Path(__file__).resolve().parents[1] / "config" / "buildings.xml"
MAP_SIZE_M = 500.0
START_REGION = (10.0, 210.0, 1.0, 10.0)
GOAL_REGION = (330.0, 490.0, 420.0, 490.0)
PLACEMENT_ATTEMPTS = 2_000
OFF_MAP_M = -9_999.0


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    z: float
    radius: float
    height: float

    @property
    def on_map(self):
        return 0 <= self.x <= MAP_SIZE_M and 0 <= self.y <= MAP_SIZE_M


@dataclass(frozen=True)
class Scene:
    level: int
    episode: int
    seed: int
    start: tuple[float, float]
    goal: tuple[float, float]
    heading_rad: float
    obstacles: tuple[Obstacle, ...]


def _load_obstacles(xml_path):
    root = ElementTree.parse(xml_path).getroot()
    if root.tag != "buildings":
        raise ValueError(f"Expected a <buildings> root in {xml_path}")

    obstacles = []
    for index, node in enumerate(root.findall("Threaten"), start=1):
        try:
            obstacle = Obstacle(*(float(node.findtext(field, "")) for field in (
                "position/x", "position/y", "position/z", "_R", "_H"
            )))
        except ValueError as error:
            raise ValueError(f"Invalid obstacle {index} in {xml_path}") from error
        values = (obstacle.x, obstacle.y, obstacle.z, obstacle.radius, obstacle.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite obstacle {index} in {xml_path}")
        if obstacle.radius <= 0 or obstacle.height <= 0:
            raise ValueError(f"Obstacle {index} in {xml_path} needs a positive radius and height")
        obstacles.append(obstacle)

    if not obstacles:
        raise ValueError(f"No obstacles in {xml_path}")
    return tuple(obstacles)


def _intersects_protected_region(obstacle):
    x, y, radius = obstacle.x, obstacle.y, obstacle.radius
    return any(
        x + radius > left and x - radius < right
        and y + radius > bottom and y - radius < top
        for left, right, bottom, top in (START_REGION, GOAL_REGION)
    )


def _load_fixed_map(xml_path):
    obstacles = _load_obstacles(xml_path)
    for index, obstacle in enumerate(obstacles):
        if not obstacle.on_map or _intersects_protected_region(obstacle):
            raise ValueError(f"Fixed map {xml_path} needs on-map centers and clear start/goal regions")
        if any(
            math.hypot(obstacle.x - other.x, obstacle.y - other.y) < obstacle.radius + other.radius
            for other in obstacles[:index]
        ):
            raise ValueError(f"Fixed map {xml_path} has overlapping obstacle footprints")
    return obstacles


def _sample_obstacle(rng, template, placed, require_separation):
    attempts = 0
    while not require_separation or attempts < PLACEMENT_ATTEMPTS:
        attempts += 1
        candidate = Obstacle(
            rng.uniform(0, MAP_SIZE_M), rng.uniform(0, MAP_SIZE_M),
            template.z, template.radius, template.height,
        )
        if _intersects_protected_region(candidate):
            continue
        if require_separation and any(
            abs(candidate.x - other.x) < candidate.radius + other.radius + 2
            and abs(candidate.y - other.y) < candidate.radius + other.radius + 2
            for other in placed
        ):
            continue
        return candidate
    return Obstacle(OFF_MAP_M, OFF_MAP_M, template.z, template.radius, template.height)


def generate_scene(level, episode, *, dimensions_xml=DEFAULT_DIMENSIONS_XML, fixed_map_xml=None):
    """Return a seeded scenario shared by all methods for a given level and episode.

    Start and goal are horizontal coordinates in metres. Sample heading, start,
    and goal first so obstacle rejection leaves those draws paired across
    levels. Level 1 accepts a caller-supplied fixed footprint XML. Levels 2 and
    3 use only the obstacle dimensions and base elevations from the canonical
    XML, with or without the conservative AABB separation rule, respectively.
    """
    if type(level) is not int or level not in (1, 2, 3):
        raise ValueError("level must be 1, 2, or 3")
    if type(episode) is not int or not 1 <= episode <= 1_000:
        raise ValueError("episode must be an integer from 1 to 1000")

    seed = 123 + episode - 1
    rng = random.Random(seed)
    heading = rng.uniform(0, math.tau)
    start = (rng.uniform(*START_REGION[:2]), rng.uniform(*START_REGION[2:]))
    goal = (rng.uniform(*GOAL_REGION[:2]), rng.uniform(*GOAL_REGION[2:]))

    if level == 1:
        if fixed_map_xml is None:
            raise ValueError("Level 1 requires a fixed non-overlapping map XML")
        obstacles = _load_fixed_map(fixed_map_xml)
    else:
        if fixed_map_xml is not None:
            raise ValueError("A fixed map XML applies only to Level 1")
        templates = _load_obstacles(dimensions_xml)
        if len(templates) != 26 or any(not 10 <= template.radius <= 50 for template in templates):
            raise ValueError("The dimensions XML must contain 26 obstacles with radii from 10 to 50 m")
        placed = []
        obstacles = []
        for template in templates:
            obstacle = _sample_obstacle(rng, template, placed, require_separation=level == 2)
            obstacles.append(obstacle)
            if obstacle.on_map:
                placed.append(obstacle)
        obstacles = tuple(obstacles)

    return Scene(level, episode, seed, start, goal, heading, obstacles)
