"""Install one shared manuscript scene in an opt-in baseline run."""

from scripts.manuscript_scenes import MAP_SIZE_M, generate_scene


def reset_manuscript_scene(env, level, episode):
    if (env.len, env.width) != (MAP_SIZE_M, MAP_SIZE_M) or len(env.Agents) != 1:
        raise ValueError("Shared maps require a 500 m by 500 m environment with one agent")

    scene = generate_scene(level, episode)
    descriptions = [
        {
            "Threaten_Type": "building",
            "position": {"x": obstacle.x, "y": obstacle.y, "z": obstacle.z},
            "_R": obstacle.radius,
            "_H": obstacle.height,
        }
        for obstacle in scene.obstacles
    ]
    buildings = [env.ThreatenFactory.Create_Threaten(row, env) for row in descriptions]
    if any(building is None for building in buildings):
        raise ValueError("The baseline obstacle factory failed to load the shared scene")

    env.reset_threads = []
    env.buildings_param = {**env.buildings_param, "Threaten": descriptions}
    env.buildings = buildings
    env.map = env.Draw_threaten_map()
    env.Agents[0].reset(scene=scene)
    return scene
