"""
改进的触觉反馈计算模块

本模块实现了改进的力反馈算法，包括：
1. 自适应力反馈计算
2. 纹理映射优化
3. 力信号滤波处理
4. 多物体接触检测

作者: [你的名字]
日期: 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from enum import Enum
from collections import deque


class TextureType(Enum):
    """纹理类型枚举"""
    ProfiledRubberSlow = 1
    CrushedRock = 2
    VenetianGranite = 3
    Smooth = 4
    Rough = 5


@dataclass
class HapticFeedbackConfig:
    """触觉反馈配置"""
    force_scale: float = 1.0          # 力缩放因子
    max_force: float = 1.0            # 最大力限制
    filter_window: int = 5            # 滤波窗口大小
    texture_intensity: float = 0.8    # 纹理强度
    contact_threshold: float = 0.01   # 接触检测阈值


class ForceFilter:
    """力信号滤波器 - 使用滑动平均滤波"""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buffer: deque = deque(maxlen=window_size)
    
    def filter(self, force: float) -> float:
        """应用滤波"""
        self.buffer.append(force)
        if len(self.buffer) < self.window_size:
            return force
        return np.mean(self.buffer)
    
    def reset(self):
        """重置滤波器"""
        self.buffer.clear()


class AdaptiveForceCalculator:
    """自适应力计算器"""
    
    def __init__(self, config: HapticFeedbackConfig):
        self.config = config
        self.force_filter = ForceFilter(config.filter_window)
        self.prev_force = 0.0
        self.force_derivative = 0.0
    
    def calculate_force(self, 
                       raw_force: float,
                       penetration_depth: float = 0.0,
                       velocity: float = 0.0) -> float:
        """
        计算自适应力反馈
        
        改进点：
        1. 考虑穿透深度
        2. 考虑接触速度
        3. 添加阻尼项
        4. 自适应缩放
        
        Args:
            raw_force: 原始力值
            penetration_depth: 穿透深度
            velocity: 接触速度
            
        Returns:
            计算后的力值
        """
        # 基础力
        base_force = raw_force * self.config.force_scale
        
        # 根据穿透深度调整力 (Hertzian接触模型简化)
        if penetration_depth > 0:
            depth_factor = np.sqrt(penetration_depth) * 0.5
            base_force += depth_factor
        
        # 添加速度阻尼
        damping_force = -0.1 * velocity
        base_force += damping_force
        
        # 限制最大力
        force = np.clip(base_force, 0.0, self.config.max_force)
        
        # 应用滤波
        filtered_force = self.force_filter.filter(force)
        
        # 计算力变化率 (用于检测冲击)
        self.force_derivative = (filtered_force - self.prev_force) / 0.001  # 假设1ms时间步
        self.prev_force = filtered_force
        
        return filtered_force
    
    def detect_contact_event(self) -> bool:
        """检测接触事件 (冲击力检测)"""
        return abs(self.force_derivative) > 100.0  # 阈值可调


class TextureMapper:
    """纹理映射器 - 优化纹理渲染"""
    
    def __init__(self, config: HapticFeedbackConfig):
        self.config = config
        self.texture_mappings: Dict[str, TextureType] = {
            "jelly_cube": TextureType.ProfiledRubberSlow,
            "cloth_sheet": TextureType.Smooth,
            "sponge": TextureType.CrushedRock,
            "liver": TextureType.VenetianGranite,
            "kidney": TextureType.ProfiledRubberSlow,
        }
        self.texture_buffers: Dict[str, deque] = {}
    
    def get_texture(self, object_name: str) -> Optional[TextureType]:
        """获取物体对应的纹理类型"""
        for key, texture in self.texture_mappings.items():
            if key in object_name.lower():
                return texture
        return None
    
    def calculate_texture_intensity(self,
                                   texture: TextureType,
                                   force: float,
                                   sliding_velocity: float = 0.0) -> float:
        """
        计算纹理强度
        
        根据力和滑动速度动态调整纹理强度
        """
        base_intensity = self.config.texture_intensity
        
        # 根据力调整强度
        force_factor = min(force * 2.0, 1.0)
        
        # 根据滑动速度调整 (滑动越快，纹理感越强)
        velocity_factor = min(abs(sliding_velocity) * 10.0, 1.0)
        
        intensity = base_intensity * (0.5 + 0.5 * force_factor) * (0.3 + 0.7 * velocity_factor)
        
        return np.clip(intensity, 0.0, 1.0)


class MultiObjectContactDetector:
    """多物体接触检测器"""
    
    def __init__(self, config: HapticFeedbackConfig):
        self.config = config
        self.active_contacts: Dict[str, Dict] = {}
    
    def detect_contacts(self, 
                       data,
                       model,
                       sensor_body_id: int) -> List[Dict]:
        """
        检测所有接触
        
        Args:
            data: MuJoCo data
            model: MuJoCo model
            sensor_body_id: 传感器物体ID
            
        Returns:
            接触列表，每个接触包含物体信息和力
        """
        contacts = []
        
        # 获取传感器物体的geom范围
        sensor_body = model.body(sensor_body_id)
        body_first_geom = sensor_body.geomadr[0]
        body_last_geom = body_first_geom + sensor_body.geomnum[0]
        
        for contact in data.contact:
            # 检查是否是传感器相关的接触
            is_contact = False
            flex_id = -1
            
            if contact.geom[0] in range(body_first_geom, body_last_geom + 1):
                is_contact = True
                flex_id = contact.flex[1] if contact.flex[1] != -1 else -1
            elif contact.geom[1] in range(body_first_geom, body_last_geom + 1):
                is_contact = True
                flex_id = contact.flex[0] if contact.flex[0] != -1 else -1
            
            if is_contact and flex_id != -1:
                # 获取flex名称
                flex_name = self._get_flex_name(model, flex_id)
                
                # 计算接触力
                contact_force = self._calculate_contact_force(contact)
                
                if contact_force > self.config.contact_threshold:
                    contacts.append({
                        'flex_id': flex_id,
                        'flex_name': flex_name,
                        'force': contact_force,
                        'position': contact.pos.copy(),
                    })
        
        return contacts
    
    def _get_flex_name(self, model, flex_id: int) -> str:
        """获取flex名称"""
        try:
            name_adr = model.name_flexadr[flex_id]
            name_binary = model.names[name_adr:]
            name_decoded = name_binary.decode()
            return name_decoded[:name_decoded.index("\0")]
        except:
            return f"flex_{flex_id}"
    
    def _calculate_contact_force(self, contact) -> float:
        """计算接触力大小"""
        # 使用接触力和摩擦力的组合
        force = np.linalg.norm(contact.frame[:3] * contact.includemargin)
        return force


class ImprovedHapticFeedback:
    """改进的触觉反馈主类"""
    
    def __init__(self, config: Optional[HapticFeedbackConfig] = None):
        self.config = config or HapticFeedbackConfig()
        self.force_calculator = AdaptiveForceCalculator(self.config)
        self.texture_mapper = TextureMapper(self.config)
        self.contact_detector = MultiObjectContactDetector(self.config)
    
    def process_contact(self,
                       hand_id: int,
                       finger: str,
                       data,
                       model,
                       raw_sensor_data: float) -> Tuple[float, Optional[TextureType]]:
        """
        处理接触并返回触觉反馈
        
        Args:
            hand_id: 手ID
            finger: 手指名称
            data: MuJoCo data
            model: MuJoCo model
            raw_sensor_data: 原始传感器数据
            
        Returns:
            (力值, 纹理类型)
        """
        # 构建传感器名称
        sensor_name = f"{'left' if hand_id == 0 else 'right'}_fingertip_{finger}"
        
        try:
            site_id = model.sensor(sensor_name).objid[0]
            sensor_body_id = model.site(site_id).bodyid[0]
        except (KeyError, IndexError):
            # 传感器不存在，使用原始数据
            force = max(0.0, raw_sensor_data / 30.0)
            return (force, None)
        
        # 检测所有接触
        contacts = self.contact_detector.detect_contacts(data, model, sensor_body_id)
        
        if not contacts:
            # 无接触
            return (0.0, None)
        
        # 找到最大接触力
        max_contact = max(contacts, key=lambda x: x['force'])
        
        # 计算力反馈
        force = self.force_calculator.calculate_force(
            max_contact['force'],
            penetration_depth=0.01,  # 简化处理
            velocity=0.0
        )
        
        # 获取纹理
        texture = self.texture_mapper.get_texture(max_contact['flex_name'])
        
        return (force, texture)
    
    def get_texture_intensity(self,
                             texture: TextureType,
                             force: float,
                             sliding_velocity: float = 0.0) -> float:
        """获取纹理强度"""
        return self.texture_mapper.calculate_texture_intensity(
            texture, force, sliding_velocity
        )
    
    def reset(self):
        """重置所有状态"""
        self.force_calculator.force_filter.reset()
        self.force_calculator.prev_force = 0.0
        self.contact_detector.active_contacts.clear()


# 默认配置
DEFAULT_HAPTIC_CONFIG = HapticFeedbackConfig(
    force_scale=1.0,
    max_force=1.0,
    filter_window=5,
    texture_intensity=0.8,
    contact_threshold=0.01
)
