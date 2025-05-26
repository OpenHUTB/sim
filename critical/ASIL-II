import json

# Global Speed Ranges
very_low_speed_range = (0, 15)
low_speed_range = (15, 50)
medium_speed_range = (50, 115)

def convert_speed_mps_to_kph(speed_mps):
    # Conversion factor: 1 km/h = 0.27778 m/s
    return speed_mps * 3.6  # 1 m/s = 3.6 km/h

with open("scenarios.json", "r") as file:
    scenarios = json.load(file)

def determine_severity(collision_type, speed_kph):

    # Determine Severity
    if collision_type == 'Pedestrian':
        if very_low_speed_range[0] <= speed_kph < very_low_speed_range[1]:
            return 'S2'
        elif low_speed_range[0] <= speed_kph < low_speed_range[1] or medium_speed_range[0] <= speed_kph < medium_speed_range[1]:
            return 'S3'
    elif collision_type == 'NPC_VEHICLE':
        if very_low_speed_range[0] <= speed_kph < very_low_speed_range[1]:
            return 'S1'
        elif low_speed_range[0] <= speed_kph < low_speed_range[1]:
            return 'S2'
        elif medium_speed_range[0] <= speed_kph < medium_speed_range[1]:
            return 'S3'
    elif collision_type == 'Obstacle':
        return 'S0'

    return 'Severity Not Defined'

def determine_exposure(weather, speed_kph):
    # Determine Exposure based on Weather and Speed
    if (weather == 'HardRainNoon' or weather == 'HardRainNight') and medium_speed_range[0] <= speed_kph < medium_speed_range[1]:
        exposure_class = 'E4'
    elif (weather == 'HardRainNoon' or weather == 'HardRainNight') and low_speed_range[0] <= speed_kph < low_speed_range[1]:
        exposure_class = 'E3'
    elif (weather == 'HardRainNoon' or weather == 'HardRainNight') and very_low_speed_range[0] <= speed_kph < very_low_speed_range[1]:
        exposure_class = 'E2'
    elif (weather == 'ClearNoon' or weather == 'ClearNight') and medium_speed_range[0] <= speed_kph < medium_speed_range[1]:
        exposure_class = 'E3'
    elif (weather == 'ClearNoon' or weather == 'ClearNight') and low_speed_range[0] <= speed_kph < low_speed_range[1]:
        exposure_class = 'E2'
    elif (weather == 'ClearNoon' or weather == 'ClearNight') and very_low_speed_range[0] <= speed_kph < very_low_speed_range[1]:
        exposure_class = 'E1'
    else:
        exposure_class = 'Exposure Not Defined'

    return exposure_class

def determine_asil(severity_class, exposure_class, controllability_class):
    # ASIL Determination based on Severity, Exposure, and Controllability
    if severity_class == 'S1' and exposure_class in ['E1', 'E2'] and controllability_class == 'C3':
        asil_class = 'QM'
    elif severity_class == 'S1' and exposure_class == 'E3' and controllability_class == 'C3':
        asil_class = 'ASIL A'
    elif severity_class == 'S1' and exposure_class == 'E4' and controllability_class == 'C3':
        asil_class = 'ASIL B'
    elif severity_class == 'S2' and exposure_class == 'E1' and controllability_class == 'C3':
        asil_class = 'QM'
    elif severity_class == 'S2' and exposure_class == 'E2' and controllability_class == 'C3':
        asil_class = 'ASIL A'
    elif severity_class == 'S2' and exposure_class == 'E3' and controllability_class == 'C3':
        asil_class = 'ASIL B'
    elif severity_class == 'S2' and exposure_class == 'E4' and controllability_class == 'C3':
        asil_class = 'ASIL C'
    elif severity_class == 'S3' and exposure_class == 'E1' and controllability_class == 'C3':
        asil_class = 'ASIL A'
    elif severity_class == 'S3' and exposure_class == 'E2' and controllability_class == 'C3':
        asil_class = 'ASIL B'
    elif severity_class == 'S3' and exposure_class == 'E3' and controllability_class == 'C3':
        asil_class = 'ASIL C'
    elif severity_class == 'S3' and exposure_class == 'E4' and controllability_class == 'C3':
        asil_class = 'ASIL D'
    else:
        asil_class = 'ASIL Not Defined'

    return asil_class


# Read scenarios from a JSON file
with open("scenarios.json", "r") as file:
    scenarios = json.load(file)

# Prepare a list to store the results
asil_results = []

for scenario in scenarios:
    # Extract necessary information from each scenario
    weather = scenario["Weather"]
    collision_type = scenario["Collision Type"]
    speed_mps = scenario["Speed at Collision"]

    # Check collision_type to decide between pedestrian, vehicle, or obstacle
    if "walker" in collision_type or "diamondback" in collision_type:
        collision_type = 'Pedestrian'
    elif "vehicle" in collision_type:
        collision_type = 'NPC_VEHICLE'
    else:
        collision_type = 'Obstacle'

    # Convert speed from m/s to kph
    speed_kph = convert_speed_mps_to_kph(speed_mps)

    # Determine Severity, Exposure, and ASIL classes
    severity_class = determine_severity(collision_type, speed_kph)
    exposure_class = determine_exposure(weather, speed_kph)
    controllability_class = 'C3'  # Assuming a default value; modify as needed
    asil_level = determine_asil(severity_class, exposure_class, controllability_class)

    # Append the results with ASIL level to the list
    scenario["ASIL Level"] = asil_level
    asil_results.append(scenario)

# Save the results to a new JSON file
with open("asil_results.json", "w") as file:
    json.dump(asil_results, file, indent=4)

print("Selected scenarios saved to:", "./asil_results.json")
