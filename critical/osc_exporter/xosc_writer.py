# osc_exporter/xosc_writer.py
# OpenSCENARIO (.xosc) 场景文件自动生成器
# 从场景配置、轨迹数据、碰撞事件生成符合 ASAM 1.2 标准的 .xosc 文件

import os
import re
import json
from datetime import datetime
from xml.dom import minidom
from xml.etree import ElementTree as ET


class XOSCWriter:
    """
    OpenSCENARIO 写入器。

    用法:
        writer = XOSCWriter()
        writer.set_scenario_config(config_dict)
        writer.set_trajectory(ego_logs, adv_logs)
        writer.set_weather(weather_dict)
        writer.add_collision_event(time_s, location)
        writer.add_pedestrian(ped_data)
        xosc_str = writer.generate()
        writer.save("results/scenarios/my_scenario.xosc")
    """

    def __init__(self, template_path=None):
        if template_path is None:
            template_path = os.path.join(os.path.dirname(__file__), "template.xosc")
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = f.read()

        # 待填充的数据
        self._context = {
            # 文件头
            "date": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "description": "CARLA Extreme Driving Scenario",
            "author": "CARLA RL Scenario Generator",

            # 参数
            "ego_speed_target": "13.89",
            "adv_speed_target": "13.89",
            "initial_distance": "20.0",

            # 地图
            "map_name": "Town10HD",

            # 自车
            "ego_model": "vehicle.tesla.model3",
            "ego_max_speed": "80.0",
            "ego_init_x": "0.0",
            "ego_init_y": "0.0",
            "ego_init_z": "0.3",
            "ego_init_h": "0.0",
            "ego_init_speed": "0.0",

            # 对抗车辆
            "adv_model": "vehicle.audi.a2",
            "adv_max_speed": "70.0",
            "adv_init_x": "20.0",
            "adv_init_y": "0.0",
            "adv_init_z": "0.3",
            "adv_init_h": "0.0",
            "adv_init_speed": "0.0",

            # 天气
            "weather_cloud": "skyClear",
            "precipitation_intensity": "0.0",
            "fog_visual_range": "100000.0",
            "date_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),

            # 触发
            "trigger_time": "3.0",
            "trigger_speed": "0.0",

            # 终止
            "collision_target": "AdvVehicle",
            "episode_timeout": "60.0",
        }

        self._pedestrians = []
        self._has_trigger = False
        self._has_pedestrian_event = False

    # ================================================================
    # 数据设置
    # ================================================================

    def set_scenario_config(self, config):
        """
        从场景配置 dict 填充基础参数。
        config 来自 scenario_configs.yaml 中的某个场景定义。
        """
        cfg = config or {}

        # 基础信息
        self._context["description"] = cfg.get("name", self._context["description"])
        self._context["initial_distance"] = str(cfg.get("initial_distance", 20.0))

        # 速度
        ego_speed = cfg.get("ego_speed", 40)
        adv_speed = cfg.get("adv_speed", 35)
        self._context["ego_speed_target"] = str(ego_speed / 3.6)
        self._context["adv_speed_target"] = str(adv_speed / 3.6)
        self._context["ego_init_speed"] = str(ego_speed / 3.6)
        self._context["adv_init_speed"] = str(adv_speed / 3.6)

        # 类别
        self._context["category"] = cfg.get("category", "")

        # 行为（急刹 / 加塞）
        behavior = cfg.get("behavior", {})
        if behavior:
            if "brake_deceleration" in behavior:
                self._context["trigger_time"] = "3.0"
                self._context["trigger_speed"] = "0.0"
                self._has_trigger = True
            if "cut_in_angle" in behavior:
                self._context["trigger_time"] = "2.0"
                self._context["trigger_speed"] = str(adv_speed / 3.6)
                self._has_trigger = True

        # 行人
        ped_cfg = cfg.get("pedestrian", {})
        if ped_cfg:
            self.add_pedestrian({
                "name": "Pedestrian01",
                "init_x": str(cfg.get("ego_speed", 40) / 3.6 * 1.5),
                "init_y": "-4.0",
                "init_h": "90.0",
                "target_x": str(cfg.get("ego_speed", 40) / 3.6 * 1.5),
                "target_y": "4.0",
                "trigger_time": str(ped_cfg.get("trigger_time", 2.0)),
                "speed": str(ped_cfg.get("cross_speed", 1.5)),
            })

        return self

    def set_weather(self, weather_dict):
        """设置天气参数"""
        if not weather_dict:
            return self
        precip = weather_dict.get("precipitation", 0)
        fog_dist = weather_dict.get("fog_distance", 0)
        fog_density = weather_dict.get("fog_density", 0)

        self._context["precipitation_intensity"] = str(precip / 100.0)
        self._context["fog_visual_range"] = (
            str(fog_dist) if fog_dist > 0 else "100000.0")
        if precip > 50:
            self._context["weather_cloud"] = "overcast"
        elif precip > 20:
            self._context["weather_cloud"] = "cloudy"
        else:
            self._context["weather_cloud"] = "skyClear"
        return self

    def set_entities(self, ego_vehicle=None, adv_vehicle=None):
        """根据实际车辆设置实体初始状态"""
        if ego_vehicle is not None:
            loc = ego_vehicle.get_location()
            vel = ego_vehicle.get_velocity()
            rot = ego_vehicle.get_transform().rotation
            self._context["ego_init_x"] = str(round(loc.x, 3))
            self._context["ego_init_y"] = str(round(loc.y, 3))
            self._context["ego_init_z"] = str(round(loc.z, 3))
            self._context["ego_init_h"] = str(round(rot.yaw, 4))
            speed = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5
            self._context["ego_init_speed"] = str(round(speed, 3))

        if adv_vehicle is not None:
            loc = adv_vehicle.get_location()
            vel = adv_vehicle.get_velocity()
            rot = adv_vehicle.get_transform().rotation
            self._context["adv_init_x"] = str(round(loc.x, 3))
            self._context["adv_init_y"] = str(round(loc.y, 3))
            self._context["adv_init_z"] = str(round(loc.z, 3))
            self._context["adv_init_h"] = str(round(rot.yaw, 4))
            speed = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5
            self._context["adv_init_speed"] = str(round(speed, 3))
        return self

    def set_trajectory(self, ego_logs, adv_logs=None):
        """
        设置车辆轨迹数据（当前版本记录为注释元数据）。
        ego_logs: list of (time, x, y, z, speed_kmh) 或更多字段
        """
        self._ego_trajectory = ego_logs or []
        self._adv_trajectory = adv_logs or []
        return self

    def add_collision_event(self, time_s, location=None):
        """记录碰撞事件"""
        self._collision_time = time_s
        self._collision_location = location
        return self

    def add_pedestrian(self, ped_data):
        """
        添加行人实体。

        ped_data: dict:
            name, init_x, init_y, init_h,
            target_x, target_y,
            trigger_time (s), speed (m/s)
        """
        self._pedestrians.append(ped_data)
        self._has_pedestrian_event = True
        return self

    def set_danger_event(self, trigger_time_s, description):
        """设置危险事件触发时间"""
        self._has_trigger = True
        self._context["trigger_time"] = str(trigger_time_s)
        self._context["description"] = description
        return self

    # ================================================================
    # 模板渲染
    # ================================================================

    def generate(self):
        """
        渲染模板为完整 .xosc XML 字符串。
        """
        content = self.template

        # 处理 pedestrians 重复段落
        ped_block = self._render_pedestrians_block()
        content = self._replace_section(content, "pedestrians", ped_block)

        # 处理 trigger_event 条件段落
        if self._has_trigger:
            trigger_block = self._render_trigger_block()
        else:
            trigger_block = ""
        content = self._replace_section(content, "trigger_event", trigger_block)

        # 处理 pedestrian_event 条件段落
        if self._has_pedestrian_event and self._pedestrians:
            ped_event_block = self._render_pedestrian_event_block()
        else:
            ped_event_block = ""
        content = self._replace_section(content, "pedestrian_event", ped_event_block)

        # 替换简单占位符
        for key, value in self._context.items():
            placeholder = "{{%s}}" % key
            content = content.replace(placeholder, str(value))

        return self._pretty_print(content)

    def _render_pedestrians_block(self):
        """渲染行人实体列表"""
        if not self._pedestrians:
            return ""
        lines = []
        for p in self._pedestrians:
            lines.append(
                '    <ScenarioObject name="%s">\n'
                '      <Pedestrian name="%s" modelName="pedestrian.adult" '
                'pedestrianCategory="pedestrian" mass="70">\n'
                '        <ParameterDeclarations/>\n'
                '      </Pedestrian>\n'
                '    </ScenarioObject>' % (p["name"], p["name"])
            )
        return "\n".join(lines)

    def _render_trigger_block(self):
        """渲染危险触发 Maneuver"""
        return (
            '        <Maneuver name="DangerManeuver">\n'
            '          <Event name="DangerEvent" priority="override">\n'
            '            <Action name="DangerAction">\n'
            '              <PrivateAction>\n'
            '                <LongitudinalAction>\n'
            '                  <SpeedAction>\n'
            '                    <SpeedActionDynamics dynamicsShape="step" '
            'value="{{trigger_speed}}" dynamicsDimension="speed"/>\n'
            '                    <SpeedActionTarget>\n'
            '                      <AbsoluteTargetSpeed value="{{trigger_speed}}"/>\n'
            '                    </SpeedActionTarget>\n'
            '                  </SpeedAction>\n'
            '                </LongitudinalAction>\n'
            '              </PrivateAction>\n'
            '            </Action>\n'
            '            <StartTrigger>\n'
            '              <ConditionGroup>\n'
            '                <Condition name="TriggerTime" delay="0" conditionEdge="rising">\n'
            '                  <ByValueCondition>\n'
            '                    <SimulationTimeCondition '
            'value="{{trigger_time}}" rule="greaterOrEqual"/>\n'
            '                  </ByValueCondition>\n'
            '                </Condition>\n'
            '              </ConditionGroup>\n'
            '            </StartTrigger>\n'
            '          </Event>\n'
            '        </Maneuver>'
        )

    def _render_pedestrian_event_block(self):
        """渲染行人横穿事件"""
        if not self._pedestrians:
            return ""
        p = self._pedestrians[0]
        return (
            '      <ManeuverGroup name="PedestrianManeuvers" maximumExecutionCount="1">\n'
            '        <Actors selectTriggeringActors="false">\n'
            '          <EntityRef entityRef="%s"/>\n'
            '        </Actors>\n'
            '        <Maneuver name="PedestrianCross">\n'
            '          <Event name="PedestrianCrossEvent" priority="overwrite">\n'
            '            <Action name="PedestrianCrossAction">\n'
            '              <PrivateAction>\n'
            '                <RoutingAction>\n'
            '                  <AcquirePositionAction>\n'
            '                    <Position>\n'
            '                      <WorldPosition x="%s" y="%s" z="0.0" h="0.0" p="0.0" r="0.0"/>\n'
            '                    </Position>\n'
            '                  </AcquirePositionAction>\n'
            '                </RoutingAction>\n'
            '              </PrivateAction>\n'
            '            </Action>\n'
            '            <StartTrigger>\n'
            '              <ConditionGroup>\n'
            '                <Condition name="PedestrianTrigger" delay="0" conditionEdge="rising">\n'
            '                  <ByValueCondition>\n'
            '                    <SimulationTimeCondition value="%s" rule="greaterOrEqual"/>\n'
            '                  </ByValueCondition>\n'
            '                </Condition>\n'
            '              </ConditionGroup>\n'
            '            </StartTrigger>\n'
            '          </Event>\n'
            '        </Maneuver>\n'
            '      </ManeuverGroup>'
            % (p["name"], p["target_x"], p["target_y"], p.get("trigger_time", "2.0"))
        )

    @staticmethod
    def _replace_section(content, section_name, replacement):
        """
        替换模板中 {{#section_name}}...{{/section_name}} 段落。

        若 replacement 为空字符串，则删除整个段落。
        """
        # 非空时：去掉标记，保留内部内容（已由调用方生成）
        start_tag = "{{#%s}}" % section_name
        end_tag = "{{/%s}}" % section_name

        # 找到标签位置
        start_idx = content.find(start_tag)
        if start_idx == -1:
            return content
        end_idx = content.find(end_tag)
        if end_idx == -1:
            return content

        # 如果 replacement 为空 → 删除整个段落
        # 如果 replacement 非空 → 用 replacement 替代标记包围区域
        if replacement:
            # 找到 start_tag 所在行的行首
            line_start = content.rfind("\n", 0, start_idx)
            if line_start == -1:
                line_start = 0
            else:
                line_start += 1

            # 找到 end_tag 所在行的行尾
            line_end = content.find("\n", end_idx)
            if line_end == -1:
                line_end = len(content)

            before = content[:line_start]
            after = content[line_end + 1:] if line_end < len(content) else ""
            return before + replacement + after
        else:
            # 删除整个段落
            line_start = content.rfind("\n", 0, start_idx)
            if line_start == -1:
                line_start = 0
            line_end = content.find("\n", end_idx)
            if line_end == -1:
                line_end = len(content) - 1
            before = content[:line_start]
            after = content[line_end + 1:] if line_end < len(content) else ""
            return before + after

    @staticmethod
    def _pretty_print(xml_string):
        """格式化 XML 输出"""
        try:
            dom = minidom.parseString(xml_string)
            return dom.toprettyxml(indent="  ", encoding="UTF-8").decode("utf-8")
        except Exception:
            return xml_string

    # ================================================================
    # 保存
    # ================================================================

    def save(self, filepath):
        """生成并保存 .xosc 文件到指定路径"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        content = self.generate()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath


# ================================================================
# 快捷函数
# ================================================================

def export_scenario(scenario, output_dir="results/scenarios"):
    """
    一键导出场景为 .xosc。

    scenario: BaseScenario 子类实例（必须已 setup）
    output_dir: 输出目录
    """
    writer = XOSCWriter()

    # 从场景实例直接获取天气参数
    if hasattr(scenario, "weather") and scenario.weather is not None:
        w = scenario.weather
        weather_dict = {
            "precipitation": getattr(w, "precipitation", 0),
            "fog_distance": getattr(w, "fog_distance", 0),
            "fog_density": getattr(w, "fog_density", 0),
        }
        writer.set_weather(weather_dict)
    writer.set_scenario_config({"name": scenario.name, "ego_speed": scenario.ego_speed_ms * 3.6})

    if scenario.ego_vehicle is not None:
        writer.set_entities(scenario.ego_vehicle, scenario.adv_vehicle)

    if scenario.logs:
        writer.set_trajectory(scenario.logs)

    filename = "%s.xosc" % scenario.name
    filepath = os.path.join(output_dir, filename)
    return writer.save(filepath)
