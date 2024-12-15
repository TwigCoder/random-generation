import streamlit as st
import numpy as np
from noise import snoise2, pnoise2
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
from datetime import datetime
import colorsys
from scipy.ndimage import gaussian_filter

NOISE_TYPES = {
    "Perlin": snoise2,
    "Simplex": pnoise2,
    "Ridged": lambda x, y, **kwargs: abs(snoise2(x, y, **kwargs)),
    "Billowed": lambda x, y, **kwargs: -abs(snoise2(x, y, **kwargs)),
    "Terraced": lambda x, y, **kwargs: np.round(snoise2(x, y, **kwargs) * 4) / 4,
    "Warped": lambda x, y, **kwargs: snoise2(
        x + snoise2(x / 2, y / 2, **kwargs),
        y + snoise2(x / 2, y / 2, **kwargs),
        **kwargs
    ),
}

BIOME_COLORS = {
    "DEEP_OCEAN": [0, 0, 139],
    "SHALLOW_OCEAN": [0, 105, 148],
    "CORAL_REEF": [255, 127, 80],
    "BEACH": [238, 214, 175],
    "SNOW": [255, 255, 255],
    "MOUNTAINS": [139, 137, 137],
    "ROCKY_MOUNTAINS": [169, 169, 169],
    "ICE_PEAKS": [220, 220, 255],
    "TUNDRA": [95, 115, 95],
    "DESERT": [255, 223, 139],
    "DESERT_HILLS": [205, 173, 89],
    "SAVANNA": [255, 195, 100],
    "RAINFOREST": [34, 139, 34],
    "TROPICAL_RAINFOREST": [0, 100, 0],
    "PLAINS": [124, 252, 0],
    "GRASSLANDS": [144, 238, 144],
    "FOREST": [160, 225, 160],
    "DENSE_FOREST": [0, 100, 0],
    "RIVER": [30, 144, 255],
    "SWAMP": [47, 79, 79],
    "MARSH": [85, 107, 47],
    "VOLCANIC": [139, 0, 0],
}

WEATHER_CONDITIONS = {
    "CLEAR": lambda x: x,
    "RAINY": lambda x: x * 0.8,
    "STORMY": lambda x: x * 0.6,
    "FOGGY": lambda x: x * 0.9 + 0.1,
    "SNOWY": lambda x: x * 0.85 + 0.15,
}


def initialize_state():
    if "last_rotation" not in st.session_state:
        st.session_state.last_rotation = datetime.now()

    if "position" not in st.session_state:
        st.session_state.position = {"x": 0, "y": 0}
        st.session_state.chunk_size = 50
        st.session_state.view_distance = 3
        st.session_state.day_cycle = 0.0
        st.session_state.show_contours = False
        st.session_state.show_rivers = False
        st.session_state.show_caves = False
        st.session_state.show_erosion = False
        st.session_state.height_multiplier = 1.0
        st.session_state.camera_x = 1.5
        st.session_state.camera_y = 1.5
        st.session_state.camera_z = 1.2
        st.session_state.terrain_rotation = 0.0
        st.session_state.weather = "CLEAR"
        st.session_state.erosion_iterations = 5
        st.session_state.temperature_variation = 0.5
        st.session_state.moisture_variation = 0.5
        st.session_state.wind_direction = 0.0
        st.session_state.wind_strength = 0.5

    if "noise_type" not in st.session_state:
        st.session_state.noise_type = "Perlin"
    if "octaves" not in st.session_state:
        st.session_state.octaves = 4
    if "persistence" not in st.session_state:
        st.session_state.persistence = 0.5
    if "lacunarity" not in st.session_state:
        st.session_state.lacunarity = 2.0
    if "seed" not in st.session_state:
        st.session_state.seed = 42


def apply_weather_and_time(terrain, weather, cycle):
    sun_position = np.sin(cycle)
    ambient_light = 0.2
    directional_light = np.maximum(sun_position * 0.6 + 0.4, ambient_light)

    weather_func = WEATHER_CONDITIONS[weather]
    terrain = weather_func(terrain)

    return terrain * directional_light


def generate_caves(height_map, threshold=0.3):
    cave_noise = np.zeros_like(height_map)
    for i in range(height_map.shape[0]):
        for j in range(height_map.shape[1]):
            cave_noise[i, j] = snoise2(i / 20, j / 20, octaves=2)

    caves = np.where(cave_noise > threshold, 0.2, 1.0)
    return caves


def apply_thermal_erosion(height_map, iterations=5, talus=0.5):
    eroded = height_map.copy()
    for _ in range(iterations):
        for i in range(1, height_map.shape[0] - 1):
            for j in range(1, height_map.shape[1] - 1):
                neighbors = [
                    eroded[i - 1, j],
                    eroded[i + 1, j],
                    eroded[i, j - 1],
                    eroded[i, j + 1],
                ]
                max_diff = max(0, eroded[i, j] - min(neighbors))
                if max_diff > talus:
                    eroded[i, j] -= max_diff * 0.5
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if (
                            0 <= i + di < height_map.shape[0]
                            and 0 <= j + dj < height_map.shape[1]
                        ):
                            eroded[i + di, j + dj] += max_diff * 0.125
    return eroded


def apply_wind_erosion(height_map, direction, strength):
    wind_vector = np.array([np.cos(direction), np.sin(direction)]) * strength
    eroded = height_map.copy()

    for i in range(1, height_map.shape[0] - 1):
        for j in range(1, height_map.shape[1] - 1):
            wind_i = int(i + wind_vector[0])
            wind_j = int(j + wind_vector[1])

            if 0 <= wind_i < height_map.shape[0] and 0 <= wind_j < height_map.shape[1]:
                diff = (height_map[i, j] - height_map[wind_i, wind_j]) * strength
                if diff > 0:
                    eroded[i, j] -= diff * 0.1
                    eroded[wind_i, wind_j] += diff * 0.1

    return eroded


def get_enhanced_biome(height, temperature, moisture):
    if height < 0.2:
        return BIOME_COLORS["DEEP_OCEAN"]
    if height < 0.3:
        if temperature > 0.7 and moisture > 0.6:
            return BIOME_COLORS["CORAL_REEF"]
        return BIOME_COLORS["SHALLOW_OCEAN"]
    if height < 0.4:
        return BIOME_COLORS["BEACH"]
    if height > 0.8:
        if temperature < 0.2:
            return BIOME_COLORS["ICE_PEAKS"]
        if temperature > 0.8:
            return BIOME_COLORS["VOLCANIC"]
        return BIOME_COLORS["SNOW"]
    if height > 0.7:
        if temperature > 0.6:
            return BIOME_COLORS["ROCKY_MOUNTAINS"]
        return BIOME_COLORS["MOUNTAINS"]
    if temperature < 0.2:
        return BIOME_COLORS["TUNDRA"]
    if temperature > 0.8:
        if height > 0.5:
            return BIOME_COLORS["DESERT_HILLS"]
        return BIOME_COLORS["DESERT"]
    if temperature > 0.6 and height < 0.5:
        return BIOME_COLORS["SAVANNA"]
    if moisture > 0.8:
        if temperature > 0.6:
            return BIOME_COLORS["TROPICAL_RAINFOREST"]
        return BIOME_COLORS["RAINFOREST"]
    if moisture > 0.6:
        if height < 0.5:
            return BIOME_COLORS["SWAMP"]
        return BIOME_COLORS["DENSE_FOREST"]
    if moisture > 0.4:
        return BIOME_COLORS["FOREST"]
    if moisture > 0.2:
        return BIOME_COLORS["GRASSLANDS"]
    return BIOME_COLORS["PLAINS"]


def main():
    st.set_page_config(layout="wide")
    initialize_state()

    col1, col2 = st.columns([4, 1])

    with col2:
        st.subheader("Map Generation")
        st.selectbox("Noise Algorithm", list(NOISE_TYPES.keys()), key="noise_type")
        st.slider("Scale", 25.0, 100.0, 50.0, key="scale")
        st.slider("Octaves", 1, 8, 4, key="octaves")
        st.slider("Persistence", 0.1, 1.0, 0.5, key="persistence")
        st.slider("Lacunarity", 1.0, 4.0, 2.0, key="lacunarity")
        st.number_input("Seed", value=42, key="seed")

        st.divider()
        st.subheader("Navigation")
        nav_col1, nav_col2, nav_col3 = st.columns(3)
        with nav_col2:
            if st.button("⬆️"):
                st.session_state.position["x"] -= 10
        with nav_col1:
            if st.button("⬅️"):
                st.session_state.position["y"] -= 10
        with nav_col3:
            if st.button("➡️"):
                st.session_state.position["y"] += 10
        with nav_col2:
            if st.button("⬇️"):
                st.session_state.position["x"] += 10

        st.divider()
        st.subheader("Environmental Controls")
        st.selectbox("Weather", list(WEATHER_CONDITIONS.keys()), key="weather")
        st.slider("Temperature Variation", 0.0, 1.0, 0.5, key="temperature_variation")
        st.slider("Moisture Variation", 0.0, 1.0, 0.5, key="moisture_variation")
        st.slider("Wind Direction (rad)", 0.0, 2 * np.pi, 0.0, key="wind_direction")
        st.slider("Wind Strength", 0.0, 1.0, 0.5, key="wind_strength")

        st.divider()
        st.subheader("Visualization Controls")
        st.checkbox("Show Contours", key="show_contours")
        st.checkbox("Show Rivers", key="show_rivers")
        st.checkbox("Show Caves", key="show_caves")
        st.checkbox("Show Erosion", key="show_erosion")
        st.slider("Time of Day", 0.0, 2 * np.pi, key="day_cycle", step=0.1)
        st.slider("Height Multiplier", 0.1, 3.0, 1.0, key="height_multiplier", step=0.1)
        st.slider("Erosion Iterations", 1, 10, 5, key="erosion_iterations")

        st.divider()
        st.subheader("Camera Controls")
        st.slider("Camera X", -2.0, 2.0, st.session_state.camera_x, key="camera_x")
        st.slider("Camera Y", -2.0, 2.0, st.session_state.camera_y, key="camera_y")
        st.slider("Camera Z", 0.1, 3.0, st.session_state.camera_z, key="camera_z")

    with col1:
        view_size = st.session_state.chunk_size * st.session_state.view_distance
        terrain = np.zeros((view_size, view_size, 3))
        height_map = np.zeros((view_size, view_size))
        temperature_map = np.zeros((view_size, view_size))
        moisture_map = np.zeros((view_size, view_size))

        noise_func = NOISE_TYPES[st.session_state.noise_type]

        for i in range(view_size):
            for j in range(view_size):
                wx = st.session_state.position["x"] + i
                wy = st.session_state.position["y"] + j

                temp_offset = np.sin(wx / 100) * st.session_state.temperature_variation
                moist_offset = np.cos(wy / 100) * st.session_state.moisture_variation

                height = noise_func(
                    wx / st.session_state.scale,
                    wy / st.session_state.scale,
                    octaves=st.session_state.octaves,
                    persistence=st.session_state.persistence,
                    lacunarity=st.session_state.lacunarity,
                    base=st.session_state.seed,
                )

                temperature = (
                    noise_func(
                        wx / st.session_state.scale,
                        wy / st.session_state.scale,
                        octaves=2,
                        persistence=0.5,
                        lacunarity=2.0,
                        base=st.session_state.seed + 1,
                    )
                    + temp_offset
                )

                moisture = (
                    noise_func(
                        wx / st.session_state.scale,
                        wy / st.session_state.scale,
                        octaves=2,
                        persistence=0.5,
                        lacunarity=2.0,
                        base=st.session_state.seed + 2,
                    )
                    + moist_offset
                )

                height_map[i, j] = height
                temperature_map[i, j] = temperature
                moisture_map[i, j] = moisture

        height_map = (height_map - height_map.min()) / (
            height_map.max() - height_map.min()
        )
        temperature_map = (temperature_map - temperature_map.min()) / (
            temperature_map.max() - temperature_map.min()
        )
        moisture_map = (moisture_map - moisture_map.min()) / (
            moisture_map.max() - moisture_map.min()
        )

        if st.session_state.show_erosion:
            height_map = apply_thermal_erosion(
                height_map, st.session_state.erosion_iterations
            )
            height_map = apply_wind_erosion(
                height_map,
                st.session_state.wind_direction,
                st.session_state.wind_strength,
            )

        if st.session_state.show_caves:
            caves = generate_caves(height_map)
            height_map = height_map * caves

        height_map = height_map * st.session_state.height_multiplier

        if st.session_state.show_rivers:
            rivers = np.zeros_like(height_map)
            flow_accumulation = np.zeros_like(height_map)

            for i in range(1, height_map.shape[0] - 1):
                for j in range(1, height_map.shape[1] - 1):
                    if height_map[i, j] > 0.7:
                        current_i, current_j = i, j
                        flow = [(current_i, current_j)]

                        while True:
                            neighbors = [
                                (current_i - 1, current_j),
                                (current_i + 1, current_j),
                                (current_i, current_j - 1),
                                (current_i, current_j + 1),
                            ]

                            min_height = height_map[current_i, current_j]
                            next_pos = None

                            for ni, nj in neighbors:
                                if (
                                    0 <= ni < height_map.shape[0]
                                    and 0 <= nj < height_map.shape[1]
                                    and height_map[ni, nj] < min_height
                                ):
                                    min_height = height_map[ni, nj]
                                    next_pos = (ni, nj)

                            if next_pos is None or height_map[next_pos] < 0.2:
                                break

                            current_i, current_j = next_pos
                            flow.append((current_i, current_j))
                            flow_accumulation[current_i, current_j] += 1

                        for fi, fj in flow:
                            if flow_accumulation[fi, fj] > 2:
                                rivers[fi, fj] = 1
                                height_map[fi, fj] *= 0.8

        for i in range(view_size):
            for j in range(view_size):
                if st.session_state.show_rivers and rivers[i, j] > 0:
                    terrain[i, j] = BIOME_COLORS["RIVER"]
                else:
                    terrain[i, j] = get_enhanced_biome(
                        height_map[i, j], temperature_map[i, j], moisture_map[i, j]
                    )

        terrain = terrain / 255.0
        terrain = apply_weather_and_time(
            terrain, st.session_state.weather, st.session_state.day_cycle
        )

        if st.session_state.show_contours:
            contours = np.zeros_like(height_map)
            contour_levels = np.linspace(0, 1, 10)
            for level in contour_levels:
                mask = np.abs(height_map - level) < 0.02
                contours[mask] = 1
            terrain[contours > 0] *= 0.7

        minimap = plt.figure(figsize=(2, 2))
        plt.imshow(terrain)
        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()

        st.image(buf, use_container_width=True)

        biome_legend = " | ".join(
            [
                "Deep Ocean",
                "Shallow Ocean",
                "Coral Reef",
                "Beach",
                "Snow",
                "Mountains",
                "Rocky Mountains",
                "Ice Peaks",
                "Tundra",
                "Desert",
                "Desert Hills",
                "Savanna",
                "Rainforest",
                "Tropical Rainforest",
                "Plains",
                "Grasslands",
                "Forest",
                "Dense Forest",
                "River",
                "Swamp",
                "Marsh",
                "Volcanic",
            ]
        )
        st.markdown(biome_legend)

        terrain_3d = go.Figure(
            data=[go.Surface(z=height_map, colorscale="earth", showscale=False)]
        )

        camera = dict(
            x=st.session_state.camera_x * np.cos(st.session_state.terrain_rotation),
            y=st.session_state.camera_y * np.sin(st.session_state.terrain_rotation),
            z=st.session_state.camera_z,
        )

        terrain_3d.update_layout(
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                camera=dict(eye=camera),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.5),
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(terrain_3d, use_container_width=True)


if __name__ == "__main__":
    main()
