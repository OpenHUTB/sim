\section{change\_lane.py}
	
	\begin{verbatim}
		import random
		import re
		import xml.etree.ElementTree as ET
		
		# Function to generate a random integer value within a range
		def random_integer(min_val, max_val):
		return random.randint(min_val, max_val)
		
		# Function to modify the variables in the Python file
		def modify_variables(original_code, original_xml, variable_ranges, num_variations):
		print("Modifying variables...")
		modified_data = []
		for i in range(num_variations):  # Change this line
		print(f"Processing variation {i + 1}...")
		modified_code = original_code
		modified_xml = original_xml
		
		for variable, value_spec in variable_ranges.items():
		if value_spec is not None:
		if "-" in value_spec:  # Range values
		min_val, max_val = map(int, value_spec.split("-"))
		modified_value = random_integer(min_val, max_val)
		elif "/" in value_spec:  # Specific values
		options = value_spec.split("/")
		modified_value = random.choice(options)
		else:  # Other specific instructions
		modified_value = value_spec
		
		# Specific condition for self._fast_vehicle_distance
		if variable == "self._fast_vehicle_distance":
		min_diff = 20
		slow_vehicle_distance_pattern = r"self._slow_vehicle_distance\s*=\s*(\d+)"
		slow_vehicle_distance_match = re.search(slow_vehicle_distance_pattern, modified_code)
		if slow_vehicle_distance_match:
		slow_vehicle_distance = int(slow_vehicle_distance_match.group(1))
		
		# Generate a modified value that meets the conditions
		while True:
		modified_value = random.choice([5, 25, 45, 65, 85])
		print(f"Modified value: {modified_value}, Slow vehicle distance: {slow_vehicle_distance}")
		if modified_value < slow_vehicle_distance and slow_vehicle_distance - modified_value >= min_diff:
		print("Condition met. Exiting loop.")
		break
		
		elif variable == "weather":
		options = ["carla.WeatherParameters.ClearNoon", "carla.WeatherParameters.HardRainNoon", "carla.WeatherParameters.ClearNight", "carla.WeatherParameters.HardRainNight"]
		modified_value = random.choice(options)
		weather_part = modified_value.split('.')[-1]
		modified_code = re.sub(r"self.output\['Weather']\s*=\s*\"\"\"",
		f"self.output['Weather'] = \"{weather_part}\"", modified_code)
		
		modified_code = re.sub(rf"\b{re.escape(variable)}\b\s*=\s*[^,\n]*", f"{variable} = {modified_value}", modified_code)
		
		modified_code = re.sub(r"self.output\['Scenario Name']\s*=\s*\"ChangeLane\"",
		f"self.output['Scenario Name'] = \"ChangeLane_{i + 1}\"", modified_code)
		
		modified_xml = original_xml.replace('type="ChangeLane"', f'type="ChangeLane_{i + 1}"')
		
		modified_data.append((modified_code, modified_xml))
		
		print("Variables modified successfully.")
		return modified_data
		
		# Read the original Python file
		with open("D:\\BaiduNetdiskDownload\\github\\scenario_runner\\srunner\\scenarios\\change_lane.py", "r") as file:
		original_code = file.read()
		
		# Read the original XML file
		with open("D:\\BaiduNetdiskDownload\\github\\scenario_runner\\srunner\\examples\\ChangeLane.xml", "r") as file:
		original_xml = file.read()
		
		# Define the variable ranges you want to change
		variable_ranges = {
			"self._fast_vehicle_velocity": "1-32",
			"self._slow_vehicle_distance": "25-160",
			"self._fast_vehicle_distance": "5/25/45/65/85",
			"self._trigger_distance": "10/20/30/40",
			"weather": "carla.WeatherParameters.ClearNoon/carla.WeatherParameters.HardRainNoon/carla.WeatherParameters.ClearNight/carla.WeatherParameters.HardRainNight",
			"desired_speed": "5-116"
		}
		
		# Generate multiple variations by modifying variables
		num_variations = 1000
		print("Generating variations...")
		modified_data = modify_variables(original_code, original_xml, variable_ranges, num_variations)
		print("Variations generated successfully.")
		
		# Write modified versions to separate files
		for i, (modified_code, modified_xml) in enumerate(modified_data):
		print(f"Writing modified files for variation {i + 1}...")
		with open(f"D:\\BaiduNetdiskDownload\\github\\scenario_runner\\srunner\\scenarios\\change_lane_{i+1}.py", "w") as code_file:
		code_file.write(modified_code)
		
		with open(f"D:\\BaiduNetdiskDownload\\github\\scenario_runner\\srunner\\examples\\ChangeLane_{i + 1}.xml", "w") as xml_file:
		xml_file.write(modified_xml)
		print(f"Modified files for variation {i + 1} written successfully.")
	\end{verbatim}
