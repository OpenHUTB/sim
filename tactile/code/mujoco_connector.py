from fingers_sim import annular_sim, index_sim, middle_sim, pinky_sim, thumb_sim
from interfaces import Engine, Visualizer
from hand import Hand
from math import radians

import mujoco as mj
import mujoco.viewer as mj_viewer

from weart import TextureType

import numpy as np
import os
from fingers_sim import *

TEXTURE_MAPPING = {
    "left kidney normal": TextureType.ProfiledRubberSlow,
    "liver cibrosis": TextureType.CrushedRock,
    "liver lesion": TextureType.VenetianGranite,
    "demo_jelly_cube": TextureType.ProfiledRubberSlow,
    "demo_jelly_ball": TextureType.VenetianGranite,
    "demo_cloth_pad": TextureType.CrushedRock,
}

class MujocoConnector(Engine):
    _plugins_loaded = False

    def __init__(self, xml_path: str, hands: tuple[Hand, Hand]):
        """Creates the MuJoCo Connector with the MJCF at the passed path.

        Args:
            xml_path (str): path to the XML file containing the MJCF
            hands (tuple[Hand, Hand]): hands configuration
        """
        if not MujocoConnector._plugins_loaded:
            try:
                if hasattr(mj, "_load_all_bundled_plugins"):
                    mj._load_all_bundled_plugins()
                else:
                    plugin_dir = os.path.join(os.path.dirname(mj.__file__), "plugin")
                    mj.mj_loadAllPluginLibraries(plugin_dir)
            finally:
                MujocoConnector._plugins_loaded = True

        spec = mj.MjSpec()
        spec = spec.from_file(xml_path)

        self._edit_hands(spec, hands)
        
        self.model = spec.compile()
        self.data = mj.MjData(self.model)

        self._fetch_hands(hands)
        self._fetch_flexes(TEXTURE_MAPPING)

        self._fetch_finger_joints(hands)
        self._init_task()

        self._should_reset = False

    def _edit_hands(self, spec: mj.MjSpec, hands: tuple[Hand, Hand]):
        for hand in hands:
            hand_body = spec.worldbody.find_child(f"{hand.side}_hand")
            if hand_body is None:
                # 如果场景中没有手部模型，跳过
                print(f"警告：场景中没有找到 {hand.side}_hand 模型")
                continue

            if hand.tracking or hand.haptics:
                controller_rotation = hand.controller_rotation if spec.compiler.degree else radians(hand.controller_rotation)
                hand_body_rotation = hand_body.alt
                hand_body_rotation.euler[1] += controller_rotation
                hand_body.alt = hand_body_rotation
            else:
                # 既不做 tracking 也不做 haptics 时，保留手模型以支持键盘/脚本控制（例如 nudge_hand）。
                # 如果你确实想隐藏未使用的手，可以在上层配置中显式删除/移除。
                pass

    def _fetch_hands(self, hands: tuple[Hand, Hand]):
        # 无论是否启用 tracking，都尝试抓取每只手的 mocap id，这样键盘 nudge 也能工作。
        self._hand_mocaps = []
        for hand in hands:
            try:
                self._hand_mocaps.append(self.model.body(f"{hand.side}_hand_mocap").mocapid[0])
            except KeyError:
                self._hand_mocaps.append(None)
    
    def _fetch_flexes(self, flex_textures: dict[str, TextureType]):
        self._flex_textures = {}
        for flex, texture in flex_textures.items():
            flex_id = self._get_flex_id(flex)
            if flex_id is not None:
                self._flex_textures[flex_id] = texture

    def _get_flex_id(self, flex_name: str):
        for id, name_adr in enumerate(self.model.name_flexadr):
            name_binary: bytes = self.model.names[name_adr:]
            name_decoded = name_binary.decode()
            name_decoded = name_decoded[:name_decoded.index("\0")]
            if name_decoded == flex_name:
                return id
        return None

    def _fetch_finger_joints(self, hands: tuple[Hand, Hand]):
        """获取手指关节ID - 修复：使用正确的joint名称，避免硬编码偏移
        
        改进点：
        1. 使用正确的joint名称而不是actuator名称
        2. 存储qpos地址避免硬编码偏移
        3. 添加错误处理
        """
        self.joint_ids = {}
        self.joint_qpos_adr = {}  # 存储qpos地址，避免硬编码偏移
        self.actuator_ids = {}    # 存储actuator ID用于控制
        
        for hand in filter(lambda h: h.tracking or h.haptics, hands):
            hand_prefix = hand.side.capitalize() + "_"
            for finger in ["Index", "Middle", "Annular", "Pinky", "Thumb"]:
                for joint_name in ["J1", "J2", "J3"]:
                    # 构建完整的joint名称，例如 "Left_Index_J1"
                    full_joint_name = hand_prefix + finger + "_" + joint_name
                    try:
                        # 获取joint的ID和qpos地址
                        joint_id = self.model.joint(full_joint_name).id
                        self.joint_ids[full_joint_name] = joint_id
                        self.joint_qpos_adr[full_joint_name] = self.model.joint(full_joint_name).qposadr[0]
                        
                        # 同时获取对应的actuator用于控制
                        actuator_name = full_joint_name + "_actuator"
                        try:
                            self.actuator_ids[full_joint_name] = self.model.actuator(actuator_name).id
                        except KeyError:
                            # 如果找不到带_actuator后缀的，尝试直接使用joint名
                            try:
                                self.actuator_ids[full_joint_name] = self.model.actuator(full_joint_name).id
                            except KeyError:
                                print(f"Warning: Actuator for {full_joint_name} not found")
                    except KeyError:
                        print(f"Warning: Joint {full_joint_name} not found in model")
                        continue
    
    def _init_task(self):
        self.middle_links = [0.030836556725044706, 0.007652464090533103, 0.021260243533355868, 0.00577589647566514, 0.008874814927647797, 0.002068895357431296]
        self.annular_links = [0.028450942497569385, 0.00801985099612217, 0.018776141709360844, 0.005880719130344506, 0.008045547312644435, 0.002237504762006266] 
        self.pinky_links = [0.019719324227772114, 0.00533552696553951, 0.013602577880681295, 0.003827636868878808, 0.005663897513197083, 0.0015979126384130552]
        self.index_links = [0.02592721506448389, 0.006917977811470749, 0.017868637273446453, 0.005247156468983898, 0.0074581157647223475, 0.001764839822760083]
        self.thumb_links = [0.0057328924064646465, 0.02297047222727691, 0.0016093799740583314, 0.007174085230947566, 0.064886404284719, 0.016350265551556033]
        self.timestep = self.model.opt.timestep
        self.maxForce = 0

    def move_hand(self, hand_id: int, position: list[float], rotation: list[float]):
        mocap_id = self._hand_mocaps[hand_id] if hand_id < len(self._hand_mocaps) else None
        if mocap_id is None:
            return
        self.data.mocap_pos[mocap_id] = position
        self.data.mocap_quat[mocap_id] = rotation

    def nudge_hand(self, hand_id: int, delta_position: list[float]):
        mocap_id = self._hand_mocaps[hand_id] if hand_id < len(self._hand_mocaps) else None
        if mocap_id is None:
            return
        self.data.mocap_pos[mocap_id] = self.data.mocap_pos[mocap_id] + np.asarray(delta_position, dtype=float)

    def _finger_value_mapping(self, closure, finger):
        max_dist = 0
        min_dist = 0
        match finger:
            case "index": 
                max_dist = 0.11
                min_dist = 0.04
            case "middle": 
                max_dist = 0.125
                min_dist = 0.03
            case "thumb":
                max_dist = 0.1
                min_dist = 0.064
            case "thumb_abd":
                max_dist = 0.0
                min_dist = -0.07
            case "annular": 
                max_dist = 0.12
                min_dist = 0.04
            case "pinky": 
                max_dist = 0.1
                min_dist = 0.045
        distance = max_dist - closure*(max_dist - min_dist)
        return distance

    def move_finger(self, hand_id: int, finger: str, closure: float, abduction: float):
        """移动手指 - 修复：正确的手指映射和调用方式
        
        改进点：
        1. 修复middle手指的错误映射
        2. 添加参数验证
        3. 使用改进的IK算法接口
        """
        # 参数验证
        if not 0 <= closure <= 1:
            print(f"Warning: closure value {closure} out of range [0, 1]")
            closure = max(0, min(1, closure))
        
        if hand_id == 1:
            hand_side = 1 
            hand = "Right"
        else: 
            hand_side = -1 
            hand = "Left"
        
        # 获取手掌位置和旋转矩阵
        palm_body_name = hand + "_Palm_" + hand
        try:
            palm = self.data.xpos[self.model.body(palm_body_name).id]
            R = self.data.xmat[self.model.body(palm_body_name).id]
            R = np.reshape(R, [3, 3])
        except KeyError:
            print(f"Warning: Palm body {palm_body_name} not found")
            return
        
        dr = self._finger_value_mapping(closure, finger)
        hand_prefix = hand + "_"
        
        # 使用改进的IK算法（如果可用），否则使用原有算法
        try:
            from fingers_sim import ik_solver
            use_improved_ik = True
        except ImportError:
            use_improved_ik = False
        
        match finger:
            case "index":
                if use_improved_ik:
                    ik_solver.move_finger_improved(
                        "index", dr, self.data, self.model, 
                        self.joint_ids, self.joint_qpos_adr, 
                        self.actuator_ids, palm, self.index_links, R, hand_prefix
                    )
                else:
                    index_sim.move_index(dr, self.data, self.model, self.joint_ids, palm, self.index_links, R, hand_prefix)
            case "middle":
                # 修复：middle只控制middle，不再错误映射到annular和pinky
                if use_improved_ik:
                    ik_solver.move_finger_improved(
                        "middle", dr, self.data, self.model,
                        self.joint_ids, self.joint_qpos_adr,
                        self.actuator_ids, palm, self.middle_links, R, hand_prefix
                    )
                else:
                    middle_sim.move_middle(dr, self.data, self.model, self.joint_ids, palm, self.middle_links, R, hand_prefix)
            case "pinky":
                if use_improved_ik:
                    ik_solver.move_finger_improved(
                        "pinky", dr, self.data, self.model,
                        self.joint_ids, self.joint_qpos_adr,
                        self.actuator_ids, palm, self.pinky_links, R, hand_prefix
                    )
                else:
                    pinky_sim.move_pinky(dr, self.data, self.model, self.joint_ids, palm, self.pinky_links, R, hand_prefix)
            case "thumb":
                abd = self._finger_value_mapping(abduction, "thumb_abd")
                if use_improved_ik:
                    ik_solver.move_thumb_improved(
                        dr, abd, self.data, self.model,
                        self.joint_ids, self.joint_qpos_adr,
                        self.actuator_ids, palm, self.thumb_links, R, hand_side
                    )
                else:
                    thumb_sim.move_thumb(dr, abd, self.data, self.model, self.joint_ids, palm, self.thumb_links, R, hand_side)
            case "annular":
                if use_improved_ik:
                    ik_solver.move_finger_improved(
                        "annular", dr, self.data, self.model,
                        self.joint_ids, self.joint_qpos_adr,
                        self.actuator_ids, palm, self.annular_links, R, hand_prefix
                    )
                else:
                    annular_sim.move_annular(dr, self.data, self.model, self.joint_ids, palm, self.annular_links, R, hand_prefix)

    def get_contact(self, hand_id: int, finger: str) -> tuple[float, TextureType | None]:
        sensor_name = "left" if hand_id == 0 else "right"
        sensor_name += "_fingertip_" + finger
        # e.g. left_fingertip_thumb

        try:
            data = self.data.sensor(sensor_name).data
        except KeyError:
            return (0.0, None)

        force = float(data[0]) / 30.0
        if force < 0:
            force = 0.0


        texture = None
        site_id = self.model.sensor(sensor_name).objid[0]
        sensor_body = self.model.body(self.model.site(site_id).bodyid[0])
        body_first_geom = sensor_body.geomadr[0]
        body_last_geom = body_first_geom + sensor_body.geomnum[0]

        for contact in self.data.contact:
            flex = -1
            geom = -1
            if contact.flex[0] != -1:
                flex = contact.flex[0]
                geom = contact.geom[1]
            elif contact.flex[1] != -1:
                flex = contact.flex[1]
                geom = contact.geom[0]

            if geom != -1:
                # means we have a contact between a flex and a geom
                if geom in range(body_first_geom, body_last_geom + 1):
                    # contact with the body of the sensor
                    if flex in self._flex_textures:
                        texture = self._flex_textures[flex]
    

        return (force, texture)

    def step_simulation(self, duration: float | None):
        if self._should_reset:
            mj.mj_resetData(self.model, self.data)
            self._should_reset = False

        if duration is None:
            mj.mj_step(self.model, self.data)
        else:
            step_count = int(duration // self.model.opt.timestep)
            mj.mj_step(self.model, self.data, nstep=step_count)
    
    def reset_simulation(self):
        self._should_reset = True

class MujocoSimpleVisualizer(Visualizer):
    def __init__(self, mujoco: MujocoConnector, framerate: int | None = None):
        self._mujoco = mujoco
        self._scene = mj.MjvScene(mujoco.model, 1000)
        self._framerate = framerate
    
    def start_visualization(self):
        print("正在启动3D可视化窗口...")
        self._viewer = mj_viewer.launch_passive(self._mujoco.model, self._mujoco.data,
                                                show_left_ui=False, show_right_ui=False)
        self._viewer.cam.azimuth = 138
        self._viewer.cam.distance = 3
        self._viewer.cam.elevation = -16
        print("3D窗口已启动！如果看不到窗口，请检查：")
        print("  1. 任务栏是否有新窗口")
        print("  2. 按 Alt+Tab 切换窗口")
        print("  3. 窗口是否被其他窗口遮挡")

    def render_frame(self):
        self._viewer.sync()

    def should_exit(self):
        return not self._viewer.is_running()
    
    def stop_visualization(self):
        self._viewer.close()
