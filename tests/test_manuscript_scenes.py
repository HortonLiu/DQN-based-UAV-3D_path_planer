import itertools
import math
import random
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from scripts.manuscript_scenes import (
    DEFAULT_DIMENSIONS_XML,
    GOAL_REGION,
    MAP_SIZE_M,
    OFF_MAP_M,
    START_REGION,
    generate_scene,
)


class ManuscriptSceneTests(unittest.TestCase):
    def test_episode_seed_and_points_are_stable_across_methods_and_levels(self):
        rng_state = random.getstate()
        first = generate_scene(2, 1)
        self.assertEqual(first, generate_scene(2, 1))
        self.assertEqual(random.getstate(), rng_state)
        self.assertEqual(first.seed, 123)
        self.assertEqual(generate_scene(3, 1_000).seed, 1122)
        self.assertNotEqual(first.start, generate_scene(2, 2).start)

        overlapping = generate_scene(3, 1)
        self.assertEqual(first.start, overlapping.start)
        self.assertEqual(first.goal, overlapping.goal)
        self.assertEqual(first.heading_rad, overlapping.heading_rad)
        self.assertGreaterEqual(first.heading_rad, 0)
        self.assertLess(first.heading_rad, math.tau)
        for (x, y), (left, right, bottom, top) in (
            (first.start, START_REGION), (first.goal, GOAL_REGION)
        ):
            self.assertTrue(left <= x <= right and bottom <= y <= top)

    def test_protected_regions_and_dimensions_apply_at_each_level(self):
        templates = ElementTree.parse(DEFAULT_DIMENSIONS_XML).getroot().findall("Threaten")
        expected = [(float(row.findtext("_R")), float(row.findtext("_H"))) for row in templates]
        self.assertEqual(len(expected), 26)
        for level in (2, 3):
            scene = generate_scene(level, 1)
            self.assertEqual([(o.radius, o.height) for o in scene.obstacles], expected)
            for obstacle in scene.obstacles:
                if not obstacle.on_map:
                    continue
                self.assertTrue(0 <= obstacle.x <= MAP_SIZE_M and 0 <= obstacle.y <= MAP_SIZE_M)
                for left, right, bottom, top in (START_REGION, GOAL_REGION):
                    self.assertTrue(
                        obstacle.x + obstacle.radius <= left or obstacle.x - obstacle.radius >= right
                        or obstacle.y + obstacle.radius <= bottom or obstacle.y - obstacle.radius >= top
                    )

    def test_level_2_separates_boxes_while_level_3_permits_overlap(self):
        separated = [obstacle for obstacle in generate_scene(2, 1).obstacles if obstacle.on_map]
        for a, b in itertools.combinations(separated, 2):
            clearance = a.radius + b.radius + 2
            self.assertTrue(abs(a.x - b.x) >= clearance or abs(a.y - b.y) >= clearance)

        overlapping = generate_scene(3, 1).obstacles
        self.assertTrue(all(obstacle.on_map for obstacle in overlapping))
        self.assertTrue(any(
            math.hypot(a.x - b.x, a.y - b.y) < a.radius + b.radius
            for a, b in itertools.combinations(overlapping, 2)
        ))

    def test_infeasible_candidate_preserves_an_off_map_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            crowded = Path(temporary) / "crowded.xml"
            tree = ElementTree.parse(DEFAULT_DIMENSIONS_XML)
            for obstacle in tree.getroot().findall("Threaten"):
                obstacle.find("_R").text = "50"
            tree.write(crowded)
            scene = generate_scene(2, 1, dimensions_xml=crowded)

        self.assertEqual(len(scene.obstacles), 26)
        off_map = [obstacle for obstacle in scene.obstacles if not obstacle.on_map]
        self.assertGreater(len(off_map), 0)
        self.assertTrue(all((obstacle.x, obstacle.y) == (OFF_MAP_M, OFF_MAP_M) for obstacle in off_map))

    def test_fixed_level_requires_supplied_valid_footprints(self):
        with self.assertRaisesRegex(ValueError, "requires a fixed"):
            generate_scene(1, 1)
        with self.assertRaisesRegex(ValueError, "Fixed map"):
            generate_scene(1, 1, fixed_map_xml=DEFAULT_DIMENSIONS_XML)

        with tempfile.TemporaryDirectory() as temporary:
            fixed = Path(temporary) / "fixed.xml"
            tree = ElementTree.parse(DEFAULT_DIMENSIONS_XML)
            root = tree.getroot()
            for obstacle in list(root)[2:]:
                root.remove(obstacle)
            for node, x in zip(root, (80, 220)):
                node.find("position/x").text = str(x)
                node.find("position/y").text = "200"
            tree.write(fixed)
            first = generate_scene(1, 1, fixed_map_xml=fixed)
            second = generate_scene(1, 2, fixed_map_xml=fixed)
            self.assertEqual(first.obstacles, second.obstacles)
            self.assertEqual([(o.x, o.y) for o in first.obstacles], [(80, 200), (220, 200)])
            self.assertNotEqual(first.start, second.start)

            root[1].find("position/x").text = "90"
            tree.write(fixed)
            with self.assertRaisesRegex(ValueError, "overlapping obstacle"):
                generate_scene(1, 1, fixed_map_xml=fixed)

    def test_out_of_protocol_arguments_are_rejected(self):
        for level, episode in ((0, 1), (2, 0), (2, 1_001), (2, 1.5)):
            with self.subTest(level=level, episode=episode), self.assertRaises(ValueError):
                generate_scene(level, episode)


if __name__ == "__main__":
    unittest.main()
