"""
数据库模块 - 用于存储实验数据和用户配置

使用SQLite作为本地数据库，存储：
1. 实验数据（力、位置、时间戳）
2. 用户配置（场景、参数）
3. 测试结果

作者: [你的名字]
日期: 2025
"""

import sqlite3
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class ExperimentData:
    """实验数据结构"""
    timestamp: datetime
    hand_id: int
    finger: str
    force: float
    texture: Optional[str]
    hand_position: List[float]
    hand_rotation: List[float]
    finger_closure: float
    finger_abduction: float
    scene_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'hand_id': self.hand_id,
            'finger': self.finger,
            'force': self.force,
            'texture': self.texture,
            'hand_position': json.dumps(self.hand_position),
            'hand_rotation': json.dumps(self.hand_rotation),
            'finger_closure': self.finger_closure,
            'finger_abduction': self.finger_abduction,
            'scene_name': self.scene_name,
        }


@dataclass
class UserConfig:
    """用户配置结构"""
    id: Optional[int] = None
    config_name: str = "default"
    scene_path: str = ""
    use_weart: bool = False
    use_vr: bool = False
    hands_config: Dict[str, Any] = None
    haptic_config: Dict[str, Any] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.hands_config is None:
            self.hands_config = {
                'left_hand': {'tracking': True, 'haptics': True},
                'right_hand': {'tracking': True, 'haptics': True},
            }
        if self.haptic_config is None:
            self.haptic_config = {
                'force_scale': 1.0,
                'max_force': 1.0,
                'texture_intensity': 0.8,
            }


class SimulationDatabase:
    """仿真数据库管理类"""
    
    def __init__(self, db_path: str = "simulation_data.db"):
        self.db_path = Path(db_path)
        self._init_database()
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_database(self):
        """初始化数据库表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 实验数据表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    hand_id INTEGER NOT NULL,
                    finger TEXT NOT NULL,
                    force REAL NOT NULL,
                    texture TEXT,
                    hand_position TEXT,
                    hand_rotation TEXT,
                    finger_closure REAL,
                    finger_abduction REAL,
                    scene_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 用户配置表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_name TEXT UNIQUE NOT NULL,
                    scene_path TEXT,
                    use_weart BOOLEAN DEFAULT 0,
                    use_vr BOOLEAN DEFAULT 0,
                    hands_config TEXT,
                    haptic_config TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 测试结果表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    test_type TEXT NOT NULL,
                    scene_name TEXT,
                    result_data TEXT,
                    metrics TEXT,
                    passed BOOLEAN,
                    duration_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiment_timestamp 
                ON experiment_data(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiment_scene 
                ON experiment_data(scene_name)
            """)
            
            conn.commit()
    
    def save_experiment_data(self, data: ExperimentData) -> int:
        """
        保存实验数据
        
        Returns:
            插入记录的ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            data_dict = data.to_dict()
            
            cursor.execute("""
                INSERT INTO experiment_data 
                (timestamp, hand_id, finger, force, texture, 
                 hand_position, hand_rotation, finger_closure, finger_abduction, scene_name)
                VALUES 
                (:timestamp, :hand_id, :finger, :force, :texture,
                 :hand_position, :hand_rotation, :finger_closure, :finger_abduction, :scene_name)
            """, data_dict)
            
            return cursor.lastrowid
    
    def save_experiment_data_batch(self, data_list: List[ExperimentData]) -> int:
        """
        批量保存实验数据
        
        Returns:
            插入记录数量
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            data_dicts = [d.to_dict() for d in data_list]
            cursor.executemany("""
                INSERT INTO experiment_data 
                (timestamp, hand_id, finger, force, texture, 
                 hand_position, hand_rotation, finger_closure, finger_abduction, scene_name)
                VALUES 
                (:timestamp, :hand_id, :finger, :force, :texture,
                 :hand_position, :hand_rotation, :finger_closure, :finger_abduction, :scene_name)
            """, data_dicts)
            
            return len(data_list)
    
    def get_experiment_data(self, 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           scene_name: Optional[str] = None,
                           hand_id: Optional[int] = None,
                           finger: Optional[str] = None,
                           limit: int = 1000) -> List[Dict[str, Any]]:
        """
        查询实验数据
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM experiment_data WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            if scene_name:
                query += " AND scene_name = ?"
                params.append(scene_name)
            if hand_id is not None:
                query += " AND hand_id = ?"
                params.append(hand_id)
            if finger:
                query += " AND finger = ?"
                params.append(finger)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def save_user_config(self, config: UserConfig) -> int:
        """保存用户配置"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO user_configs 
                (config_name, scene_path, use_weart, use_vr, 
                 hands_config, haptic_config, updated_at)
                VALUES 
                (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                config.config_name,
                config.scene_path,
                config.use_weart,
                config.use_vr,
                json.dumps(config.hands_config),
                json.dumps(config.haptic_config),
            ))
            
            return cursor.lastrowid
    
    def get_user_config(self, config_name: str = "default") -> Optional[UserConfig]:
        """获取用户配置"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM user_configs WHERE config_name = ?
            """, (config_name,))
            
            row = cursor.fetchone()
            if row:
                return UserConfig(
                    id=row['id'],
                    config_name=row['config_name'],
                    scene_path=row['scene_path'],
                    use_weart=bool(row['use_weart']),
                    use_vr=bool(row['use_vr']),
                    hands_config=json.loads(row['hands_config']),
                    haptic_config=json.loads(row['haptic_config']),
                    created_at=datetime.fromisoformat(row['created_at']),
                )
            return None
    
    def save_test_result(self,
                        test_name: str,
                        test_type: str,
                        scene_name: str,
                        result_data: Dict[str, Any],
                        metrics: Dict[str, float],
                        passed: bool,
                        duration_ms: int) -> int:
        """保存测试结果"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO test_results 
                (test_name, test_type, scene_name, result_data, metrics, passed, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                test_name,
                test_type,
                scene_name,
                json.dumps(result_data),
                json.dumps(metrics),
                passed,
                duration_ms,
            ))
            
            return cursor.lastrowid
    
    def get_test_results(self, 
                        test_type: Optional[str] = None,
                        scene_name: Optional[str] = None,
                        passed: Optional[bool] = None) -> List[Dict[str, Any]]:
        """获取测试结果"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM test_results WHERE 1=1"
            params = []
            
            if test_type:
                query += " AND test_type = ?"
                params.append(test_type)
            if scene_name:
                query += " AND scene_name = ?"
                params.append(scene_name)
            if passed is not None:
                query += " AND passed = ?"
                params.append(passed)
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                result = dict(row)
                result['result_data'] = json.loads(result['result_data'])
                result['metrics'] = json.loads(result['metrics'])
                results.append(result)
            
            return results
    
    def get_statistics(self, scene_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取统计数据
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 基础统计
            query = "SELECT COUNT(*) as total, AVG(force) as avg_force, MAX(force) as max_force FROM experiment_data"
            params = []
            
            if scene_name:
                query += " WHERE scene_name = ?"
                params.append(scene_name)
            
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            stats = {
                'total_records': row['total'],
                'avg_force': row['avg_force'],
                'max_force': row['max_force'],
            }
            
            # 按手指统计
            finger_query = """
                SELECT finger, COUNT(*) as count, AVG(force) as avg_force
                FROM experiment_data
            """
            if scene_name:
                finger_query += " WHERE scene_name = ?"
                cursor.execute(finger_query + " GROUP BY finger", params)
            else:
                cursor.execute(finger_query + " GROUP BY finger")
            
            finger_stats = {}
            for row in cursor.fetchall():
                finger_stats[row['finger']] = {
                    'count': row['count'],
                    'avg_force': row['avg_force'],
                }
            stats['by_finger'] = finger_stats
            
            return stats
    
    def export_to_csv(self, filepath: str, 
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None):
        """导出数据到CSV文件"""
        import csv
        
        data = self.get_experiment_data(start_time, end_time, limit=100000)
        
        if not data:
            print("No data to export")
            return
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Exported {len(data)} records to {filepath}")
    
    def clear_old_data(self, days: int = 30):
        """清理旧数据"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM experiment_data 
                WHERE created_at < datetime('now', '-{} days')
            """.format(days))
            
            deleted = cursor.rowcount
            print(f"Cleared {deleted} old records")
            return deleted


# 全局数据库实例
db = SimulationDatabase()
