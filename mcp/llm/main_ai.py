#!/usr/bin/env python3
"""
FastMCP GitHub Assistant - 使用FastMCP框架的智能GitHub助手
集成Deepseek AI模型，支持自然语言查询GitHub仓库
使用FastMCP装饰器方式实现MCP工具调用机制
"""

import sys
import json
import re
from pathlib import Path
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uvicorn
import aiohttp
from typing import Optional

# CARLA module - optional, only needed for simulator control
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    carla = None
    print("WARNING: carla module not found. CARLA features will be disabled.")
    print("To install carla, run: pip install D:/hutb/PythonAPI/carla/dist/hutb-2.9.16-cp310-cp310-win_amd64.whl")

# 添加src目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from fastmcp import FastMCP
from src.github_client import GitHubClient
from src.config import config
from src.utils.logger import app_logger

# 创建FastMCP实例
mcp = FastMCP("AI智能助手")


class CarlaClient:
    """CARLA客户端封装类"""

    def __init__(self):
        self.client = None
        self.world = None
        self.actors = []

    async def connect(self, host='localhost', port=2000):
        """连接CARLA服务器"""
        if not CARLA_AVAILABLE:
            app_logger.error("❌ CARLA模块未安装，无法连接服务器")
            return False
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(10)
            self.world = self.client.get_world()
            app_logger.info("✅ CARLA服务器连接成功")
            return True
        except Exception as e:
            app_logger.error(f"❌ 连接CARLA失败: {str(e)}")
            return False

    async def spawn_vehicle(self, vehicle_type='model3'):
        """生成车辆"""
        try:
            blueprint = self.world.get_blueprint_library().find(f'vehicle.tesla.{vehicle_type}')
            spawn_point = self.world.get_map().get_spawn_points()[0]
            vehicle = self.world.spawn_actor(blueprint, spawn_point)
            self.actors.append(vehicle)
            app_logger.info(f"🚗 生成车辆: {vehicle_type}")

            return vehicle
        except Exception as e:
            app_logger.error(f"❌ 生成车辆失败: {str(e)}")
            return None

    async def set_weather(self, weather_type='clear'):
        """设置天气"""
        if not CARLA_AVAILABLE or not self.world:
            app_logger.error("❌ CARLA未连接，无法设置天气")
            return False
        weather_presets = {
            'clear': carla.WeatherParameters(
                cloudiness=0, precipitation=0, precipitation_deposits=0,
                wind_intensity=10, sun_azimuth_angle=0, sun_altitude_angle=75,
                fog_density=0, fog_distance=0, wetness=0
            ),
            'rain': carla.WeatherParameters(
                cloudiness=100, precipitation=80, precipitation_deposits=50,
                wind_intensity=30, sun_azimuth_angle=0, sun_altitude_angle=15,
                fog_density=10, fog_distance=100, wetness=60
            ),
            'fog': carla.WeatherParameters(
                cloudiness=80, precipitation=0, precipitation_deposits=0,
                wind_intensity=5, sun_azimuth_angle=0, sun_altitude_angle=30,
                fog_density=90, fog_distance=50, wetness=20
            )
        }
        if weather_type in weather_presets:
            self.world.set_weather(weather_presets[weather_type])
            return True
        return False

    async def get_traffic_lights(self):
        """获取交通灯状态"""
        if not CARLA_AVAILABLE or not self.world:
            return []
        lights = [light for light in self.world.get_actors() if 'traffic_light' in light.type_id]
        return lights[:5]  # 只返回前5个

    async def spawn_pedestrian(self, pedestrian_type='walker'):
        """生成行人"""
        try:
            # 查找行人蓝图
            blueprint_library = self.world.get_blueprint_library()
            walker_blueprints = blueprint_library.filter('walker.pedestrian.*')
            if not walker_blueprints:
                app_logger.error("❌ 未找到行人蓝图")
                return None
            
            blueprint = walker_blueprints[0]  # 使用第一个找到的行人蓝图
            spawn_point = self.world.get_map().get_spawn_points()[0]
            # 设置随机位置偏移，避免与车辆重叠
            spawn_point.location.x += 5.0
            
            pedestrian = self.world.spawn_actor(blueprint, spawn_point)
            self.actors.append(pedestrian)
            app_logger.info(f"🚶 生成行人: {pedestrian_type}")
            
            # 将视角对准生成的行人
            self.set_spectator_view(pedestrian)
            return pedestrian
        except Exception as e:
            app_logger.error(f"❌ 生成行人失败: {str(e)}")
            return None

    def set_spectator_view(self, target_actor):
        """将视角对准目标actor"""
        try:
            spectator = self.world.get_spectator()
            target_transform = target_actor.get_transform()
            
            # 设置相机位置在目标actor前方5米，上方2米处
            # 这样可以从正面看到行人
            camera_location = carla.Location(
                x=target_transform.location.x + 5.0,  # 前方5米
                y=target_transform.location.y,
                z=target_transform.location.z + 2.0
            )
            
            # 计算相机朝向，指向行人
            # yaw=180.0 让相机朝向行人方向
            camera_rotation = carla.Rotation(
                pitch=-15.0,  # 略微向下看
                yaw=180.0,    # 朝向行人
                roll=0.0
            )
            
            camera_transform = carla.Transform(camera_location, camera_rotation)
            spectator.set_transform(camera_transform)
            app_logger.info(f"👁️  视角已对准actor {target_actor.id}")
            return True
        except Exception as e:
            app_logger.error(f"❌ 设置视角失败: {str(e)}")
            return False

    async def get_vehicle_blueprints(self):
        """获取可用车辆蓝图列表"""
        if not CARLA_AVAILABLE or not self.world:
            return []
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        result = list(set([bp.id for bp in blueprints]))
        result.sort()
        return result[:20]

    async def set_vehicle_autopilot(self, vehicle_id, enabled):
        """设置车辆自动驾驶"""
        if not CARLA_AVAILABLE or not self.world:
            return False
        try:
            vehicle = self.world.get_actor(vehicle_id)
            if vehicle:
                vehicle.set_autopilot(enabled)
                return True
            return False
        except Exception as e:
            app_logger.error(f"❌ 设置自动驾驶失败: {str(e)}")
            return False

    async def apply_vehicle_control(self, vehicle_id, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        """应用车辆控制"""
        if not CARLA_AVAILABLE or not self.world:
            return False
        try:
            vehicle = self.world.get_actor(vehicle_id)
            if vehicle:
                throttle = max(0.0, min(1.0, throttle))
                steer = max(-1.0, min(1.0, steer))
                brake = max(0.0, min(1.0, brake))
                control = carla.VehicleControl(
                    throttle=throttle,
                    steer=steer,
                    brake=brake,
                    hand_brake=hand_brake,
                    reverse=reverse
                )
                vehicle.apply_control(control)
                return True
            return False
        except Exception as e:
            app_logger.error(f"❌ 应用车辆控制失败: {str(e)}")
            return False

    async def get_vehicle_state(self, vehicle_id):
        """获取车辆状态"""
        if not CARLA_AVAILABLE or not self.world:
            return None
        try:
            vehicle = self.world.get_actor(vehicle_id)
            if vehicle:
                transform = vehicle.get_transform()
                velocity = vehicle.get_velocity()
                speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                return {
                    "vehicle_id": vehicle_id,
                    "type_id": vehicle.type_id,
                    "location": {
                        "x": round(transform.location.x, 2),
                        "y": round(transform.location.y, 2),
                        "z": round(transform.location.z, 2)
                    },
                    "rotation": {
                        "pitch": round(transform.rotation.pitch, 2),
                        "yaw": round(transform.rotation.yaw, 2),
                        "roll": round(transform.rotation.roll, 2)
                    },
                    "speed_kmh": round(speed, 2),
                }
            return None
        except Exception as e:
            app_logger.error(f"❌ 获取车辆状态失败: {str(e)}")
            return None

    async def destroy_vehicle(self, vehicle_id):
        """销毁车辆"""
        if not CARLA_AVAILABLE or not self.world:
            return False
        try:
            vehicle = self.world.get_actor(vehicle_id)
            if vehicle:
                vehicle.destroy()
                self.actors = [a for a in self.actors if a.id != vehicle_id]
                return True
            return False
        except Exception as e:
            app_logger.error(f"❌ 销毁车辆失败: {str(e)}")
            return False

    async def cleanup(self):
        """清理环境"""
        for actor in self.actors:
            if actor.is_alive:
                actor.destroy()
        self.actors = []
        app_logger.info("🧹 清理所有CARLA actor")


# 全局CARLA客户端实例
carla_client = CarlaClient()

async def connect_carla_impl(host: str = 'localhost', port: int = 2000) -> str:
    """（实际功能：连接CARLA服务器）"""
    if not CARLA_AVAILABLE:
        return "❌ CARLA模块未安装。请运行: pip install D:/hutb/PythonAPI/carla/dist/hutb-2.9.16-cp310-cp310-win_amd64.whl"
    success = await carla_client.connect(host, port)
    return "✅ CARLA服务器连接成功" if success else "❌ 连接CARLA服务器失败"


async def spawn_vehicle_impl(query: str, **kwargs) -> str:
    """（实际功能：生成车辆）"""
    vehicle = await carla_client.spawn_vehicle(query)
    if vehicle:
        return f"✅ 已生成车辆: {query} (ID: {vehicle.id})"
    return "❌ 车辆生成失败"


async def set_weather_impl(owner: str, repo: str) -> str:
    """（实际功能：设置天气）"""
    weather_types = {'clear': '晴天', 'rain': '雨天', 'fog': '雾天'}
    success = await carla_client.set_weather(repo.lower())
    return f"✅ 天气已设置为 {weather_types.get(repo.lower(), repo)}" if success else "❌ 不支持的天气类型"


async def get_traffic_lights_impl(query: str, **kwargs) -> str:
    """（实际功能：获取交通灯信息）"""
    if not CARLA_AVAILABLE:
        return "❌ CARLA模块未安装，无法获取交通灯信息"
    lights = await carla_client.get_traffic_lights()
    if not lights:
        return "❌ 未找到交通灯或CARLA未连接"
    result = ["🚦 交通灯状态:"]
    for i, light in enumerate(lights, 1):
        state = "绿色" if light.state == carla.TrafficLightState.Green else \
            "红色" if light.state == carla.TrafficLightState.Red else \
                "黄色"
        result.append(f"{i}. {light.type_id} - {state} (位置: {light.get_location()})")
    return "\n".join(result)


async def cleanup_scene_impl(**kwargs) -> str:
    """（实际功能：清理环境）"""
    await carla_client.cleanup()
    return "✅ 已清理所有车辆和物体"


async def spawn_pedestrian_impl(query: str, **kwargs) -> str:
    """（实际功能：生成行人）"""
    pedestrian = await carla_client.spawn_pedestrian(query)
    if pedestrian:
        return f"✅ 已生成行人: {query} (ID: {pedestrian.id})"
    return "❌ 行人生成失败"


async def get_vehicle_blueprints_impl(**kwargs) -> str:
    """（实际功能：获取车辆蓝图列表）"""
    blueprints = await carla_client.get_vehicle_blueprints()
    if blueprints:
        return "可用车辆蓝图:\n" + "\n".join(f"  - {bp}" for bp in blueprints)
    return "❌ 无法获取车辆蓝图，请检查CARLA连接"


async def set_vehicle_autopilot_impl(vehicle_id: int, enabled: bool, **kwargs) -> str:
    """（实际功能：设置自动驾驶）"""
    success = await carla_client.set_vehicle_autopilot(vehicle_id, enabled)
    if success:
        status = "开启" if enabled else "关闭"
        return f"✅ 车辆 {vehicle_id} 自动驾驶已{status}"
    return f"❌ 设置车辆 {vehicle_id} 自动驾驶失败"


async def apply_vehicle_control_impl(vehicle_id: int, throttle: float = 0.0, steer: float = 0.0, brake: float = 0.0, hand_brake: bool = False, reverse: bool = False, **kwargs) -> str:
    """（实际功能：应用车辆控制）"""
    success = await carla_client.apply_vehicle_control(vehicle_id, throttle, steer, brake, hand_brake, reverse)
    if success:
        return f"✅ 车辆 {vehicle_id} 控制已应用: 油门={throttle}, 转向={steer}, 刹车={brake}"
    return f"❌ 车辆 {vehicle_id} 控制应用失败"


async def get_vehicle_state_impl(vehicle_id: int, **kwargs) -> str:
    """（实际功能：获取车辆状态）"""
    state = await carla_client.get_vehicle_state(vehicle_id)
    if state:
        return f"✅ 车辆 {vehicle_id} 状态:\n  位置: ({state['location']['x']}, {state['location']['y']}, {state['location']['z']})\n  速度: {state['speed_kmh']} km/h\n  车型: {state['type_id']}"
    return f"❌ 获取车辆 {vehicle_id} 状态失败"


async def destroy_vehicle_impl(vehicle_id: int, **kwargs) -> str:
    """（实际功能：销毁车辆）"""
    success = await carla_client.destroy_vehicle(vehicle_id)
    if success:
        return f"✅ 车辆 {vehicle_id} 已销毁"
    return f"❌ 销毁车辆 {vehicle_id} 失败"


# ============ FastMCP 工具装饰器版本 ============

@mcp.tool()
async def connect_carla(host: str = 'localhost', port: int = 2000) -> str:
    """（实际功能：连接CARLA）"""
    return await connect_carla_impl(host, port)


@mcp.tool()
async def spawn_vehicle(query: str, language: Optional[str] = None,
                                     sort: str = "stars", limit: int = 8) -> str:
    """（实际功能：生成车辆）"""
    return await spawn_vehicle_impl(query)


@mcp.tool()
async def set_weather(owner: str, repo: str) -> str:
    """（实际功能：设置天气）"""
    return await set_weather_impl(owner, repo)


@mcp.tool()
async def get_traffic_lights(query: str, user_type: Optional[str] = None) -> str:
    """（实际功能：获取交通灯）"""
    return await get_traffic_lights_impl(query)


@mcp.tool()
async def cleanup_scene(language: Optional[str] = None, period: str = "daily") -> str:
    """（实际功能：清理环境）"""
    return await cleanup_scene_impl()


@mcp.tool()
async def spawn_pedestrian(query: str, user_type: Optional[str] = None) -> str:
    """（实际功能：生成行人）"""
    return await spawn_pedestrian_impl(query)



# ============ AI助手类（集成Deepseek AI） ============

class FastMCPGitHubAssistant:
    """FastMCP GitHub AI助手 - 集成Deepseek AI与FastMCP工具"""

    def __init__(self):
        # 将FastMCP工具转换为标准MCP工具格式供AI使用
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "connect_carla",
                    "description": "连接CARLA服务器",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "host": {"type": "string", "description": "CARLA服务器地址", "default": "localhost"},
                            "port": {"type": "integer", "description": "CARLA服务器端口", "default": 2000}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "spawn_vehicle",
                    "description": "生成指定类型的车辆（如model3, a2等）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "车辆型号", "enum": ["model3", "a2", "mustang"]}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_weather",
                    "description": "设置天气（clear/rain/fog）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "description": "固定值weather"},
                            "repo": {"type": "string", "enum": ["clear", "rain", "fog"]}
                        },
                        "required": ["owner", "repo"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_traffic_lights",
                    "description": "获取交通灯状态",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "固定值traffic"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "cleanup_scene",
                    "description": "清理仿真环境",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "spawn_pedestrian",
                    "description": "生成行人",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "行人类型，默认为walker"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_vehicle_blueprints",
                    "description": "获取所有可用的车辆蓝图列表",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_vehicle_autopilot",
                    "description": "设置车辆自动驾驶模式（开启或关闭）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vehicle_id": {"type": "integer", "description": "车辆ID"},
                            "enabled": {"type": "boolean", "description": "是否开启自动驾驶"}
                        },
                        "required": ["vehicle_id", "enabled"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_vehicle_control",
                    "description": "手动控制车辆（油门、转向、刹车等）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vehicle_id": {"type": "integer", "description": "车辆ID"},
                            "throttle": {"type": "number", "description": "油门（0.0-1.0）", "default": 0.0},
                            "steer": {"type": "number", "description": "转向（-1.0到1.0，负值左转正值右转）", "default": 0.0},
                            "brake": {"type": "number", "description": "刹车（0.0-1.0）", "default": 0.0},
                            "hand_brake": {"type": "boolean", "description": "手刹", "default": False},
                            "reverse": {"type": "boolean", "description": "倒车", "default": False}
                        },
                        "required": ["vehicle_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_vehicle_state",
                    "description": "获取车辆当前状态（位置、速度、朝向等）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vehicle_id": {"type": "integer", "description": "车辆ID"}
                        },
                        "required": ["vehicle_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "destroy_vehicle",
                    "description": "销毁指定车辆",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vehicle_id": {"type": "integer", "description": "车辆ID"}
                        },
                        "required": ["vehicle_id"]
                    }
                }
            },
        ]

    def process_markdown(self, text):
        """在Python端处理Markdown格式"""
        result = text

        # 处理标题
        result = re.sub(r'^### (.+)$', r'<h3><strong>\1</strong></h3>', result, flags=re.MULTILINE)
        result = re.sub(r'^## (.+)$', r'<h2><strong>\1</strong></h2>', result, flags=re.MULTILINE)
        result = re.sub(r'^# (.+)$', r'<h1><strong>\1</strong></h1>', result, flags=re.MULTILINE)

        # 处理粗体链接 **[text](url)**
        result = re.sub(r'\*\*\[([^\]]+)\]\(([^)]+)\)\*\*', r'<strong><a href="\2" target="_blank">\1</a></strong>',
                        result)

        # 处理普通链接 [text](url)
        result = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', result)

        # 处理粗体文本 **text**
        result = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', result)

        # 处理换行
        result = result.replace('\n', '<br>')

        return result

    async def call_deepseek_with_tools(self, messages):
        """调用Deepseek API，包含FastMCP工具定义"""
        headers = config.get_deepseek_headers()

        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "tools": self.tools,
            "tool_choice": "auto",
            "max_tokens": 2000,
            "temperature": 0.7
        }

        app_logger.debug(f"📤 调用Deepseek API，消息数: {len(messages)}")
        app_logger.debug(f"📤 最后一条消息: {json.dumps(messages[-1], ensure_ascii=False)[:200]}")

        async with aiohttp.ClientSession() as session:
            async with session.post(config.DEEPSEEK_API_URL, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    app_logger.debug(f"📥 API响应状态: {response.status}")
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"].get("content", "")
                        app_logger.debug(f"📥 响应内容长度: {len(content) if content else 0}")
                        app_logger.debug(f"📥 响应内容预览: {content[:100] if content else 'None'}")
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"Deepseek API调用失败: {response.status} - {error_text}")

    async def execute_fastmcp_tool_call(self, tool_call):
        """执行FastMCP工具调用 - 桥接到FastMCP装饰器函数"""
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])

        app_logger.info(f"🔧 执行FastMCP工具: {function_name}")
        app_logger.info(f"📝 参数: {arguments}")

        try:
            # 调用实际的工具实现函数（避免FastMCP装饰器问题）
            if function_name == "connect_carla":
                result = await connect_carla_impl(
                    host=arguments.get("host", "localhost"),
                    port=arguments.get("port", 2000)
                )
                return {
                    "success": True,
                    "data": result
                }

            elif function_name == "spawn_vehicle":
                app_logger.info(f"🚗 spawn_vehicle参数详情: {arguments}")
                result = await spawn_vehicle_impl(
                    query=arguments["query"]
                )
                app_logger.info(f"🚗 spawn_vehicle执行结果: {result}")
                return {
                    "success": True,
                    "data": result
                }

            elif function_name == "set_weather":
                result = await set_weather_impl(
                    owner=arguments["owner"],
                    repo=arguments["repo"]
                )
                return {
                    "success": True,
                    "data": result
                }

            elif function_name == "get_traffic_lights":
                result = await get_traffic_lights_impl(
                    query=arguments["query"],
                    user_type=arguments.get("user_type")
                )
                return {
                    "success": True,
                    "data": result
                }

            elif function_name == "cleanup_scene":
                result = await cleanup_scene_impl(
                    language=arguments.get("language"),
                    period=arguments.get("period", "daily")
                )
                return {
                    "success": True,
                    "data": result
                }
            
            elif function_name == "spawn_pedestrian":
                result = await spawn_pedestrian_impl(
                    query=arguments["query"],
                    user_type=arguments.get("user_type")
                )
                return {
                    "success": True,
                    "data": result
                }

            elif function_name == "get_vehicle_blueprints":
                result = await get_vehicle_blueprints_impl()
                return {
                    "success": True,
                    "data": result
                }

            elif function_name == "set_vehicle_autopilot":
                result = await set_vehicle_autopilot_impl(
                    vehicle_id=arguments["vehicle_id"],
                    enabled=arguments["enabled"]
                )
                return {
                    "success": True,
                    "data": result
                }

            elif function_name == "apply_vehicle_control":
                result = await apply_vehicle_control_impl(
                    vehicle_id=arguments["vehicle_id"],
                    throttle=arguments.get("throttle", 0.0),
                    steer=arguments.get("steer", 0.0),
                    brake=arguments.get("brake", 0.0),
                    hand_brake=arguments.get("hand_brake", False),
                    reverse=arguments.get("reverse", False)
                )
                return {
                    "success": True,
                    "data": result
                }

            elif function_name == "get_vehicle_state":
                result = await get_vehicle_state_impl(
                    vehicle_id=arguments["vehicle_id"]
                )
                return {
                    "success": True,
                    "data": result
                }

            elif function_name == "destroy_vehicle":
                result = await destroy_vehicle_impl(
                    vehicle_id=arguments["vehicle_id"]
                )
                return {
                    "success": True,
                    "data": result
                }
            elif function_name == "search_github_repositories":
                result = await search_github_repositories_impl(
                    query=arguments["query"],
                    language=arguments.get("language"),
                    sort=arguments.get("sort", "stars"),
                    limit=arguments.get("limit", 8)
                )
                return {
                    "success": True,
                    "data": result
                }

            elif function_name == "get_repository_details":
                result = await get_repository_details_impl(
                    owner=arguments["owner"],
                    repo=arguments["repo"]
                )
                return {
                    "success": True,
                    "data": result
                }

            elif function_name == "search_github_users":
                result = await search_github_users_impl(
                    query=arguments["query"],
                    user_type=arguments.get("user_type")
                )
                return {
                    "success": True,
                    "data": result
                }

            elif function_name == "get_trending_repositories":
                result = await get_trending_repositories_impl(
                    language=arguments.get("language"),
                    period=arguments.get("period", "daily")
                )
                return {
                    "success": True,
                    "data": result
                }
            else:
                return {
                    "success": False,
                    "error": f"未知的工具: {function_name}"
                }

        except Exception as e:
            app_logger.error(f"❌ FastMCP工具执行失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def chat(self, user_message):
        """处理聊天请求 - 使用FastMCP工具的AI对话"""
        # 初始消息
        messages = [
            {
                "role": "system",
                "content": """你是一个CARLA仿真器智能助手，基于FastMCP框架提供服务。你有以下工具可以使用：

CARLA仿真功能：
1. connect_carla - 连接CARLA服务器（默认localhost:2000）
2. spawn_vehicle - 生成车辆（model3/a2/mustang）
3. spawn_pedestrian - 生成行人
4. set_weather - 设置天气（clear/rain/fog）
5. get_traffic_lights - 查看交通灯状态
6. cleanup_scene - 清理仿真场景
7. get_vehicle_blueprints - 获取可用车辆蓝图列表
8. set_vehicle_autopilot - 开启/关闭车辆自动驾驶
9. apply_vehicle_control - 手动控制车辆（油门/转向/刹车/倒车）
10. get_vehicle_state - 获取车辆状态（位置、速度等）
11. destroy_vehicle - 销毁车辆

车辆控制说明：
- 开启自动驾驶：set_vehicle_autopilot(vehicle_id=车辆ID, enabled=true)
- 关闭自动驾驶：set_vehicle_autopilot(vehicle_id=车辆ID, enabled=false)
- 加速：apply_vehicle_control(vehicle_id=车辆ID, throttle=0.5)
- 刹车：apply_vehicle_control(vehicle_id=车辆ID, brake=1.0)
- 左转：apply_vehicle_control(vehicle_id=车辆ID, steer=-0.5)
- 右转：apply_vehicle_control(vehicle_id=车辆ID, steer=0.5)
- 倒车：apply_vehicle_control(vehicle_id=车辆ID, reverse=true, throttle=0.3)
- 查看状态：get_vehicle_state(vehicle_id=车辆ID)

通用策略：
- 必须先连接CARLA服务器才能使用CARLA相关功能
- 不要自动连接CARLA服务器，只在用户明确要求时连接
- 生成车辆后会返回车辆ID，后续操作需要用到这个ID
- 手动控制时注意参数范围：throttle/brake 0-1，steer -1到1
- 如果用户说"加速"、"踩油门"等，使用apply_vehicle_control增加throttle
- 如果用户说"刹车"、"停车"等，使用apply_vehicle_control增加brake
- 如果用户说"自动驾驶"、"自动行驶"等，使用set_vehicle_autopilot

用户指令示例：
- "连接carla服务器" -> connect_carla(host="localhost", port=2000)
- "生成一辆model3" -> spawn_vehicle(query="model3")
- "开启自动驾驶" -> set_vehicle_autopilot(vehicle_id=车辆ID, enabled=true)
- "油门踩到0.8" -> apply_vehicle_control(vehicle_id=车辆ID, throttle=0.8)
- "刹车" -> apply_vehicle_control(vehicle_id=车辆ID, brake=1.0)
- "左转" -> apply_vehicle_control(vehicle_id=车辆ID, steer=-0.5)
- "查看车辆状态" -> get_vehicle_state(vehicle_id=车辆ID)
- "设置雨天" -> set_weather(owner="weather", repo="rain")
- "查看交通灯" -> get_traffic_lights(query="traffic")
- "清理场景" -> cleanup_scene()
- "销毁车辆" -> destroy_vehicle(vehicle_id=车辆ID)

本助手基于FastMCP框架构建，提供高效、类型安全的工具调用体验。"""
            },
            {"role": "user", "content": user_message}
        ]

        # 第一次API调用
        app_logger.info(f"💬 用户消息: {user_message}")
        response = await self.call_deepseek_with_tools(messages)
        assistant_message = response["choices"][0]["message"]

        # 检查是否有工具调用
        tool_calls = assistant_message.get("tool_calls", [])
        messages.append(assistant_message)

        # 执行FastMCP工具调用 - 支持多轮调用
        max_iterations = 5  # 最多5轮工具调用，防止无限循环
        iteration = 0
        all_tool_calls = []
        
        while tool_calls and iteration < max_iterations:
            iteration += 1
            app_logger.info(f"🔧 第{iteration}轮：检测到 {len(tool_calls)} 个FastMCP工具调用")
            all_tool_calls.extend(tool_calls)

            for tool_call in tool_calls:
                app_logger.info(f"🔨 执行FastMCP工具: {tool_call['function']['name']}")
                tool_result = await self.execute_fastmcp_tool_call(tool_call)
                app_logger.info(f"✅ FastMCP工具执行完成，结果长度: {len(str(tool_result))}")

                # 添加工具结果到消息历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })

            # 再次调用API获取回答或下一轮工具调用
            app_logger.info(f"🤖 第{iteration}轮：正在生成回答...")
            app_logger.debug(f"� 发送给API的消息数量: {len(messages)}")
            app_logger.debug(f"📤 最后一条消息类型: {messages[-1].get('role')}")
            
            try:
                next_response = await self.call_deepseek_with_tools(messages)
                app_logger.debug(f"📥 API响应: {json.dumps(next_response, ensure_ascii=False, indent=2)}")
                
                next_message = next_response["choices"][0]["message"]
                messages.append(next_message)
                
                # 检查是否还有工具调用
                tool_calls = next_message.get("tool_calls", [])
                
                if not tool_calls:
                    # 没有更多工具调用，返回最终结果
                    final_content = next_message.get("content", "")
                    app_logger.info(f"✅ 最终回答生成成功，长度: {len(final_content)}")
                    app_logger.debug(f"📝 最终回答内容: {final_content}")

                    if not final_content or final_content.strip() == "":
                        app_logger.warning("❌ 警告：最终回答为空")
                        final_content = "操作已完成。"

                    return {
                        "message": self.process_markdown(final_content),
                        "tool_calls": all_tool_calls,
                        "conversation": messages
                    }
                else:
                    app_logger.info(f"🔄 检测到更多工具调用，继续执行...")
                    
            except Exception as e:
                app_logger.error(f"❌ 生成回答时出错: {str(e)}")
                import traceback
                app_logger.error(f"❌ 错误堆栈: {traceback.format_exc()}")
                return {
                    "message": f"工具调用过程中出错: {str(e)}",
                    "tool_calls": all_tool_calls,
                    "conversation": messages
                }
        
        if iteration >= max_iterations:
            app_logger.warning(f"⚠️ 达到最大迭代次数 {max_iterations}，停止工具调用")
            return {
                "message": "操作已执行，但可能未完全完成（达到最大调用次数）。",
                "tool_calls": all_tool_calls,
                "conversation": messages
            }
        
        # 没有工具调用的情况
        return {
            "message": self.process_markdown(assistant_message["content"]),
            "tool_calls": None,
            "conversation": messages
        }


# ============ FastAPI Web界面（AI对话版） ============

app = FastAPI(title="FastMCP GitHub Assistant")


def get_web_interface():
    """生成AI对话Web界面HTML"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FastMCP GitHub Assistant - AI智能助手</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            * { 
                margin: 0; 
                padding: 0; 
                box-sizing: border-box; 
            }

            body {
                font-family: 'Segoe UI', 'Microsoft YaHei', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                line-height: 1.6;
            }

            .container {
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }

            .header {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                padding: 12px 20px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 15px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.18);
            }

            .header h1 {
                color: #2d3748;
                font-size: 1.5em;
                margin: 0;
                font-weight: 700;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .chat-container {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 20px;
                flex: 1;
                display: flex;
                flex-direction: column;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.18);
            }

            .messages {
                flex: 1;
                overflow-y: auto;
                overflow-x: hidden;
                padding: 15px;
                margin-bottom: 15px;
                background: rgba(248, 250, 252, 0.5);
                border-radius: 15px;
                border: 1px solid rgba(226, 232, 240, 0.5);
                height: calc(100vh - 280px);
                min-height: 400px;
                max-height: calc(100vh - 280px);
                scroll-behavior: smooth;
            }

            .message {
                margin-bottom: 15px;
                padding: 15px 20px;
                border-radius: 15px;
                max-width: 85%;
                word-wrap: break-word;
                position: relative;
                animation: messageSlide 0.3s ease-out;
            }

            @keyframes messageSlide {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .user-message {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                margin-left: auto;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                border-bottom-right-radius: 5px;
            }

            .assistant-message {
                background: linear-gradient(135deg, #f8fafc, #e2e8f0);
                color: #2d3748;
                margin-right: auto;
                border-left: 4px solid #667eea;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
                border-bottom-left-radius: 5px;
            }

            .tools-used {
                background: rgba(102, 126, 234, 0.05);
                margin-top: 10px;
                border-radius: 10px;
                font-size: 0.9em;
                border: 1px solid rgba(102, 126, 234, 0.2);
                overflow: hidden;
            }

            .tools-header {
                background: rgba(102, 126, 234, 0.1);
                padding: 10px 12px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: space-between;
                font-weight: 600;
                color: #667eea;
                transition: all 0.3s ease;
            }

            .tools-header:hover {
                background: rgba(102, 126, 234, 0.15);
            }

            .tools-toggle {
                font-size: 0.9em;
                transition: all 0.3s ease;
                font-weight: bold;
            }

            .tools-content {
                padding: 12px;
                display: none;
                border-top: 1px solid rgba(102, 126, 234, 0.1);
            }

            .tools-content.show {
                display: block;
            }

            .input-form {
                display: flex;
                gap: 12px;
                align-items: flex-end;
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.9));
                padding: 15px;
                border-radius: 15px;
                border: 1px solid rgba(102, 126, 234, 0.2);
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(10px);
            }

            .message-input {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid transparent;
                border-radius: 12px;
                background: white;
                font-size: 0.95em;
                resize: none;
                min-height: 44px;
                max-height: 120px;
                transition: all 0.3s ease;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                font-family: inherit;
                line-height: 1.4;
            }

            .message-input:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15), 0 4px 15px rgba(0, 0, 0, 0.15);
                transform: translateY(-1px);
            }

            .message-input::placeholder {
                color: #9ca3af;
                font-style: italic;
            }

            .send-button {
                width: 44px;
                height: 44px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                border: none;
                border-radius: 50%;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
                position: relative;
            }

            .send-button i {
                color: white;
                font-size: 16px;
            }

            .send-button:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
                background: linear-gradient(135deg, #5a67d8, #6b46c1);
            }

            .send-button:active:not(:disabled) {
                transform: translateY(0px);
                box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
            }

            .send-button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
                background: linear-gradient(135deg, #9ca3af, #6b7280);
            }

            .loading {
                display: none;
                text-align: center;
                padding: 25px;
                margin: 15px 0;
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                border-radius: 15px;
                border: 1px solid rgba(102, 126, 234, 0.2);
            }

            .loading.show { 
                display: block; 
            }

            .loading-content {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 15px;
            }

            .loading-text {
                color: #667eea;
                font-weight: 600;
                font-size: 1.2em;
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .loading-spinner {
                width: 24px;
                height: 24px;
                border: 3px solid rgba(102, 126, 234, 0.2);
                border-top: 3px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }

            .example-questions {
                background: linear-gradient(135deg, rgba(248, 250, 252, 0.8), rgba(241, 245, 249, 0.8));
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 15px;
                border: 1px solid rgba(226, 232, 240, 0.5);
                backdrop-filter: blur(5px);
            }

            .welcome-message {
                color: #4a5568;
                margin-bottom: 15px;
                font-size: 1em;
                line-height: 1.5;
                text-align: center;
                padding: 15px;
                background: rgba(255, 255, 255, 0.6);
                border-radius: 12px;
                border-left: 4px solid #667eea;
            }

            .example-questions h3 {
                color: #2d3748;
                margin-bottom: 15px;
                font-size: 1em;
                text-align: center;
                font-weight: 600;
            }

            .examples-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
            }

            .example-item {
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(248, 250, 252, 0.9));
                border-radius: 10px;
                padding: 12px 16px;
                cursor: pointer;
                transition: all 0.3s ease;
                border-left: 3px solid #667eea;
                font-size: 0.9em;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                border: 1px solid rgba(226, 232, 240, 0.3);
                text-align: center;
            }

            .example-item:hover {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                transform: translateY(-2px) scale(1.02);
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }

            .assistant-message h1 {
                font-size: 1.4em;
                color: #2d3748;
                margin: 15px 0 10px 0;
                font-weight: 700;
            }

            .assistant-message h2 {
                font-size: 1.2em;
                color: #2d3748;
                margin: 12px 0 8px 0;
                font-weight: 600;
            }

            .assistant-message h3 {
                font-size: 1.1em;
                color: #2d3748;
                margin: 10px 0 6px 0;
                font-weight: 600;
            }

            /* 响应式设计 */
            @media (max-width: 768px) {
                .container {
                    padding: 10px;
                }

                .header h1 {
                    font-size: 1.5em;
                }

                .message {
                    max-width: 95%;
                    padding: 12px 15px;
                }

                .examples-grid {
                    grid-template-columns: 1fr;
                    gap: 8px;
                }

                .input-form {
                    flex-direction: column;
                    gap: 12px;
                    padding: 12px;
                }

                .message-input {
                    min-height: 40px;
                }

                .send-button {
                    width: 100%;
                    height: 44px;
                }

                .messages {
                    height: calc(100vh - 320px);
                }
            }

            /* 滚动条美化 */
            .messages::-webkit-scrollbar {
                width: 6px;
            }

            .messages::-webkit-scrollbar-track {
                background: rgba(226, 232, 240, 0.3);
                border-radius: 3px;
            }

            .messages::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #667eea, #764ba2);
                border-radius: 3px;
            }

            .messages::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #5a67d8, #6b46c1);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 人形机器人智能助手</h1>
            </div>

            <div class="chat-container">
                <div class="messages" id="messages">
                    <div class="example-questions">
                        <div class="welcome-message">
                            👋 欢迎使用基于FastMCP框架的人形机器人智能助手！集成GitHub搜索 + CARLA仿真控制。
                            <br><br>
                            🔧 <strong>技术特色</strong>：本助手使用FastMCP装饰器实现工具定义，提供类型安全、自动化的MCP体验！
                        </div>
                        <h3>💡 试试这些问题：</h3>
                        <div class="examples-grid">
                            <div class="example-item" onclick="askExample('连接CARLA仿真服务器')">
                            🔗 连接服务器
                            </div>
                            <div class="example-item" onclick="askExample('设置雨天天气条件')">
                                🌫️ 天气设置（默认雨天）
                            </div>
                        </div>
                    </div>
                </div>

                <div class="loading" id="loading">
                    <div class="loading-content">
                        <div class="loading-text">
                            <div class="loading-spinner"></div>
                            <span>FastMCP工具调用中...</span>
                        </div>
                    </div>
                </div>

                <form class="input-form" onsubmit="return submitForm(event)">
                    <textarea 
                        id="messageInput" 
                        class="message-input" 
                        placeholder="问我任何人形机器人相关问题，我会使用FastMCP工具来帮你搜索..."
                        rows="2"
                        onkeydown="handleKeyPress(event)"
                    ></textarea>
                    <button type="submit" class="send-button" id="sendButton">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </form>
            </div>
        </div>

<script>
function askExample(text) {
    document.getElementById('messageInput').value = text;
    submitMessage();
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        submitMessage();
    }
}

function submitForm(event) {
    event.preventDefault();
    submitMessage();
    return false;
}

async function submitMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    if (!message) return;

    addMessage(message, 'user');
    input.value = '';
    showLoading(true);

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: 'message=' + encodeURIComponent(message)
        });

        if (response.ok) {
            const result = await response.json();
            addMessage(result.message, 'assistant', result.tool_calls);
        } else {
            addMessage('抱歉，发生了错误，请稍后重试。', 'assistant');
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('网络连接错误，请检查网络后重试。', 'assistant');
    } finally {
        showLoading(false);
    }
}

function addMessage(content, sender, toolCalls) {
    const messages = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    let html = `<div>${content}</div>`;

    if (toolCalls && toolCalls.length > 0) {
        const toolsId = 'tools-' + Date.now();
        html += `
            <div class="tools-used">
                <div class="tools-header" onclick="toggleTools('${toolsId}')">
                    <span>🔧 使用的FastMCP工具 (${toolCalls.length}个)</span>
                    <span class="tools-toggle" id="toggle-${toolsId}">▼</span>
                </div>
                <div class="tools-content" id="${toolsId}">`;

        for (let i = 0; i < toolCalls.length; i++) {
            const tool = toolCalls[i];
            const args = JSON.parse(tool.function.arguments);
            let argStr = '';
            for (const k in args) {
                if (argStr) argStr += ', ';
                argStr += `${k}: "${args[k]}"`;
            }
            html += `<div>• <strong>@mcp.tool() ${tool.function.name}</strong>(${argStr})</div>`;
        }

        html += `
                </div>
            </div>`;
    }

    messageDiv.innerHTML = html;
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

function toggleTools(toolsId) {
    const content = document.getElementById(toolsId);
    const toggle = document.getElementById('toggle-' + toolsId);

    if (content.classList.contains('show')) {
        content.classList.remove('show');
        toggle.classList.remove('expanded');
        toggle.textContent = '▼';
    } else {
        content.classList.add('show');
        toggle.classList.add('expanded');
        toggle.textContent = '▲';
    }
}

function showLoading(show) {
    const loading = document.getElementById('loading');
    const sendButton = document.getElementById('sendButton');

    if (show) {
        loading.classList.add('show');
        sendButton.disabled = true;
    } else {
        loading.classList.remove('show');
        sendButton.disabled = false;
    }
}
</script>
    </body>
    </html>
    """
    return html_content


@app.get("/", response_class=HTMLResponse)
async def index():
    """主页面 - AI对话界面"""
    return get_web_interface()


@app.post("/chat")
async def chat(message: str = Form(...)):
    """处理聊天请求 - 使用FastMCP工具的AI对话"""
    try:
        result = await assistant.chat(message)
        return {
            "success": True,
            "message": result["message"],
            "tool_calls": result["tool_calls"]
        }
    except Exception as e:
        app_logger.error(f"❌ FastMCP聊天处理失败: {str(e)}")
        return {
            "success": False,
            "message": f"抱歉，处理您的请求时出现错误: {str(e)}",
            "tool_calls": None
        }


# 创建全局AI助手实例
assistant = FastMCPGitHubAssistant()


def main():
    """主函数 - 可以选择启动Web界面或MCP服务器"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        # 启动MCP服务器模式
        print("[MCP] 启动FastMCP AI助手MCP服务器...")

        # 验证配置
        if not config.validate():
            print("[ERROR] 配置验证失败")
            print("[INFO] 请确保环境变量包含:")
            print("[INFO] 请确保 .env 文件包含以下必要配置：")
            print("   - GITHUB_TOKEN=your_github_token")
            print("   - DEEPSEEK_API_KEY=your_deepseek_api_key")
            return

        print("[OK] 配置验证通过")
        print("[TOOLS] 已注册MCP工具:")
        print("   - search_github_repositories")
        print("   - get_repository_details")
        print("   - search_github_users")
        print("   - get_trending_repositories")
        print("[READY] 等待AI连接...")

        # 启动FastMCP服务器
        mcp.run()
    else:
        # 默认启动Web AI对话界面
        print("[WEB] 启动FastMCP AI助手对话界面...")
        print("[AI] 集成Deepseek AI + FastMCP工具")

        # 验证配置
        if not config.validate():
            print("[ERROR] 配置验证失败，请检查环境变量设置")
            print("[INFO] 请确保 .env 文件包含以下必要配置：")
            print("   - GITHUB_TOKEN=your_github_token")
            print("   - DEEPSEEK_API_KEY=your_deepseek_api_key")
            return

        print("[OK] 配置验证通过")
        print("[TOOLS] FastMCP工具已注册:")
        print("   - @mcp.tool() search_github_repositories")
        print("   - @mcp.tool() get_repository_details")
        print("   - @mcp.tool() search_github_users")
        print("   - @mcp.tool() get_trending_repositories")
        print("[URL] 访问地址: http://localhost:3000")
        print("[INFO] 基于FastMCP框架 + Deepseek AI智能对话")
        print()

        uvicorn.run(app, host="localhost", port=3000)


if __name__ == "__main__":
    main()
