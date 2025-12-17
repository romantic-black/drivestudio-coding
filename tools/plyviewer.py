"""
PLY文件查看器工具类
使用nerfstudio自带的viewer在web浏览器中渲染和显示PLY点云文件

使用示例:
    1. 命令行使用:
        python tools/plyviewer.py path/to/pointcloud.ply
        python tools/plyviewer.py path/to/pointcloud1.ply path/to/pointcloud2.ply --port 7007
    
    2. Python代码使用:
        from tools.plyviewer import PLYViewer
        
        viewer = PLYViewer(host="0.0.0.0", port=7007)
        viewer.start_viewer()
        viewer.load_and_display("path/to/pointcloud.ply")
        viewer.wait()  # 保持运行直到Ctrl+C
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, List, Tuple
import numpy as np
import open3d as o3d
import viser
from rich.console import Console

# nerfstudio viewer的缩放比例（用于坐标缩放）
VISER_NERFSTUDIO_SCALE_RATIO: float = 10.0

CONSOLE = Console()

# 全局viewer实例管理
_global_viewer: Optional['PLYViewer'] = None


class PLYViewer:
    """PLY点云文件查看器，使用nerfstudio viewer在web浏览器中显示"""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        point_size: float = 0.01,
        point_shape: str = "circle",
        auto_fallback: bool = True,
    ):
        """
        初始化PLY查看器
        
        Args:
            host: Web服务器主机地址，默认为"0.0.0.0"（监听所有接口）
            port: Web服务器端口，如果为None则自动选择可用端口
            point_size: 点云中每个点的大小
            point_shape: 点的形状，"circle"或"square"
            auto_fallback: 如果指定端口被占用，是否自动查找可用端口（默认True）
        """
        self.host = host
        self.port = port
        self.point_size = point_size
        self.point_shape = point_shape
        self.auto_fallback = auto_fallback  # 如果端口被占用，是否自动查找可用端口
        self.viser_server: Optional[viser.ViserServer] = None
        self.viewer_info: Optional[str] = None
        self._loaded_point_clouds: List[str] = []  # 记录已加载的点云名称
    
    @classmethod
    def get_or_create_global_viewer(
        cls,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        point_size: float = 0.01,
        point_shape: str = "circle",
        auto_fallback: bool = True,
    ) -> 'PLYViewer':
        """
        获取或创建全局viewer实例（单例模式）
        用于在同一个viewer中添加多个点云
        
        Args:
            host: Web服务器主机地址
            port: Web服务器端口，如果为None则自动选择可用端口
            point_size: 点云中每个点的大小
            point_shape: 点的形状，"circle"或"square"
            auto_fallback: 如果指定端口被占用，是否自动查找可用端口
            
        Returns:
            全局PLYViewer实例
        """
        global _global_viewer
        
        if _global_viewer is None:
            _global_viewer = cls(
                host=host,
                port=port,
                point_size=point_size,
                point_shape=point_shape,
                auto_fallback=auto_fallback,
            )
            CONSOLE.log("[cyan]Created global PLYViewer instance[/cyan]")
        else:
            # 如果已存在全局viewer，检查是否需要启动
            if _global_viewer.viser_server is None:
                _global_viewer.start_viewer()
            CONSOLE.log("[cyan]Using existing global PLYViewer instance[/cyan]")
        
        return _global_viewer
    
    @classmethod
    def reset_global_viewer(cls) -> None:
        """重置全局viewer实例（用于测试或重新开始）"""
        global _global_viewer
        if _global_viewer is not None:
            _global_viewer.stop()
        _global_viewer = None
        CONSOLE.log("[yellow]Global PLYViewer instance reset[/yellow]")
    
    def _is_safe_to_kill(self, pid: int) -> bool:
        """
        检查进程是否安全可以kill（只kill可能是viewer的进程）
        
        Args:
            pid: 进程ID
            
        Returns:
            True if safe to kill, False otherwise
        """
        import subprocess
        
        # 尝试使用psutil（如果可用）
        try:
            import psutil
            try:
                proc = psutil.Process(pid)
                cmdline = ' '.join(proc.cmdline()).lower()
                
                # 只kill包含viser或viewer相关关键词的Python进程
                # 避免kill掉notebook、jupyter、cursor等重要进程
                safe_keywords = ['viser', 'viewer', 'plyviewer']
                dangerous_keywords = ['jupyter', 'notebook', 'ipykernel', 'cursor', 'ssh']
                
                # 检查是否包含危险关键词
                if any(keyword in cmdline for keyword in dangerous_keywords):
                    return False
                
                # 检查是否是Python进程且包含安全关键词
                if 'python' in cmdline and any(keyword in cmdline for keyword in safe_keywords):
                    return True
                    
                return False
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return False
        except ImportError:
            # 如果没有psutil，使用更保守的方法
            try:
                # 使用ps命令检查（Linux）
                result = subprocess.run(
                    ['ps', '-p', str(pid), '-o', 'cmd='],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    cmdline = result.stdout.strip().lower()
                    dangerous_keywords = ['jupyter', 'notebook', 'ipykernel', 'cursor', 'ssh']
                    if any(keyword in cmdline for keyword in dangerous_keywords):
                        return False
                    # 如果是Python进程且包含viewer相关关键词，认为是安全的
                    if 'python' in cmdline and ('viewer' in cmdline or 'viser' in cmdline):
                        return True
                return False
            except (FileNotFoundError, subprocess.TimeoutExpired):
                # 如果无法检查，为了安全起见，不kill
                return False
    
    def _kill_port_process(self, port: int) -> bool:
        """
        安全地关闭占用指定端口的进程（只kill可能是viewer的进程）
        
        Args:
            port: 要释放的端口号
            
        Returns:
            True if successfully killed a process, False otherwise
        """
        import subprocess
        import signal
        
        try:
            # 尝试使用lsof查找占用端口的进程
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                killed_any = False
                for pid in pids:
                    try:
                        pid_int = int(pid.strip())
                        # 检查是否安全可以kill
                        if not self._is_safe_to_kill(pid_int):
                            CONSOLE.log(f"[yellow]Skipping process {pid_int} on port {port} (not safe to kill)[/yellow]")
                            continue
                        
                        # 检查进程是否存在
                        os.kill(pid_int, 0)  # 检查进程是否存在
                        # 尝试优雅关闭
                        os.kill(pid_int, signal.SIGTERM)
                        CONSOLE.log(f"[yellow]Sent SIGTERM to viewer process {pid_int} on port {port}[/yellow]")
                        killed_any = True
                    except (ProcessLookupError, ValueError, PermissionError) as e:
                        # 进程不存在或没有权限
                        continue
                
                if killed_any:
                    # 等待一下让进程关闭
                    import time
                    time.sleep(0.5)
                    # 检查是否还有进程占用端口
                    result_check = subprocess.run(
                        ['lsof', '-ti', f':{port}'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result_check.returncode == 0 and result_check.stdout.strip():
                        # 如果还有进程，检查是否安全后再kill
                        pids = result_check.stdout.strip().split('\n')
                        for pid in pids:
                            try:
                                pid_int = int(pid.strip())
                                if self._is_safe_to_kill(pid_int):
                                    os.kill(pid_int, signal.SIGKILL)
                                    CONSOLE.log(f"[yellow]Force killed viewer process {pid_int} on port {port}[/yellow]")
                            except (ProcessLookupError, ValueError, PermissionError):
                                pass
                        time.sleep(0.2)
                    return True
                return False
            else:
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            CONSOLE.log(f"[yellow]Could not check processes on port {port}: {e}[/yellow]")
            return False
        
    def _get_free_port(self, default_port: int = 7007) -> int:
        """获取一个可用的端口号"""
        import socket
        if self.port is not None:
            # 如果指定了端口，检查是否可用
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('', self.port))
                sock.close()
                return self.port
            except OSError:
                # 端口被占用
                if self.auto_fallback:
                    # 尝试安全地kill占用端口的viewer进程
                    CONSOLE.log(f"[yellow]Port {self.port} is already in use. Attempting to free the port (only viewer processes)...[/yellow]")
                    if self._kill_port_process(self.port):
                        # 再次尝试绑定
                        import time
                        time.sleep(0.5)  # 等待端口释放
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        try:
                            sock.bind(('', self.port))
                            sock.close()
                            CONSOLE.log(f"[green]Port {self.port} is now available[/green]")
                            return self.port
                        except OSError:
                            pass  # 如果还是不行，继续fallback
                    
                    # 如果kill失败或端口仍被占用，自动查找可用端口
                    CONSOLE.log(f"[yellow]Could not free port {self.port} (may be used by other processes). Automatically finding an available port...[/yellow]")
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.bind(('', 0))
                    port = sock.getsockname()[1]
                    sock.close()
                    CONSOLE.log(f"[green]Using port {port} instead. Access viewer at: http://localhost:{port}[/green]")
                    return port
                else:
                    # 不自动fallback，尝试安全kill进程
                    CONSOLE.log(f"[yellow]Port {self.port} is already in use. Attempting to free the port (only viewer processes)...[/yellow]")
                    if self._kill_port_process(self.port):
                        import time
                        time.sleep(0.5)
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        try:
                            sock.bind(('', self.port))
                            sock.close()
                            CONSOLE.log(f"[green]Port {self.port} is now available[/green]")
                            return self.port
                        except OSError:
                            pass
                    # 如果无法释放端口，抛出异常
                    raise RuntimeError(
                        f"Port {self.port} is already in use and could not be freed safely. "
                        f"The port may be used by other processes (notebook, cursor, etc.). "
                        f"Please manually stop the viewer or use auto_fallback=True to use a different port."
                    )
        
        # 尝试使用默认端口
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('', default_port))
            sock.close()
            return default_port
        except OSError:
            # 如果默认端口被占用，找一个可用端口
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('', 0))
            port = sock.getsockname()[1]
            sock.close()
            return port
    
    def load_ply(self, ply_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载PLY文件
        
        Args:
            ply_path: PLY文件路径
            
        Returns:
            points: (N, 3) 点云坐标数组
            colors: (N, 3) 点云颜色数组，值范围[0, 255]
        """
        ply_path = Path(ply_path)
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")
        
        CONSOLE.log(f"[cyan]Loading PLY file: {ply_path}[/cyan]")
        
        # 使用open3d读取PLY文件
        pcd = o3d.io.read_point_cloud(str(ply_path))
        
        if len(pcd.points) == 0:
            raise ValueError(f"PLY file contains no points: {ply_path}")
        
        # 转换为numpy数组
        points = np.asarray(pcd.points)
        
        # 处理颜色
        if pcd.has_colors():
            # open3d的颜色范围是[0, 1]，转换为[0, 255]
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        else:
            # 如果没有颜色，使用默认颜色（白色）
            colors = np.ones((len(points), 3), dtype=np.uint8) * 255
        
        CONSOLE.log(f"[green]✓ Loaded {len(points)} points[/green]")
        CONSOLE.log(f"  Point range: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
                    f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
                    f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        
        return points, colors
    
    def start_viewer(self) -> None:
        """启动viewer服务器"""
        if self.viser_server is not None:
            CONSOLE.log("[yellow]Viewer already started[/yellow]")
            return
        
        port = self._get_free_port()
        self.port = port  # 保存实际使用的端口
        CONSOLE.log(f"[cyan]Starting viewer server on {self.host}:{port}[/cyan]")
        
        self.viser_server = viser.ViserServer(host=self.host, port=port)
        
        if self.host == "0.0.0.0":
            self.viewer_info = f"Viewer running locally at: http://localhost:{port} (listening on 0.0.0.0)"
        else:
            self.viewer_info = f"Viewer running locally at: http://{self.host}:{port}"
        
        CONSOLE.log(f"[green]{self.viewer_info}[/green]")
        CONSOLE.log("[yellow]Press Ctrl+C to stop the viewer[/yellow]")
    
    def add_point_cloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        name: str = "/point_cloud",
        visible: bool = True,
    ) -> None:
        """
        添加点云到viewer
        
        Args:
            points: (N, 3) 点云坐标数组
            colors: (N, 3) 点云颜色数组，值范围[0, 255]。如果为None，使用默认颜色
            name: 点云在场景树中的名称
            visible: 是否默认可见
        """
        if self.viser_server is None:
            raise RuntimeError("Viewer not started. Call start_viewer() first.")
        
        # 应用nerfstudio的缩放比例
        scaled_points = points * VISER_NERFSTUDIO_SCALE_RATIO
        
        # 处理颜色
        # viser的add_point_cloud接受单个颜色元组(r, g, b)或颜色数组(N, 3)
        # 颜色值范围应该是[0, 255]的整数
        if colors is None:
            # 使用默认颜色（白色）
            colors_final = (255, 255, 255)
        elif isinstance(colors, np.ndarray):
            # 确保颜色值在[0, 255]范围内
            colors_uint8 = np.clip(colors, 0, 255).astype(np.uint8)
            # 如果所有点颜色相同，使用单个颜色元组（更高效）
            if len(colors_uint8.shape) == 2 and colors_uint8.shape[1] == 3:
                if len(colors_uint8) > 0 and np.allclose(colors_uint8, colors_uint8[0]):
                    colors_final = tuple(colors_uint8[0].tolist())
                else:
                    # 使用颜色数组 (N, 3)
                    colors_final = colors_uint8
            else:
                colors_final = tuple(colors_uint8.tolist())
        else:
            colors_final = colors
        
        # 添加点云到viewer
        self.viser_server.add_point_cloud(
            name=name,
            points=scaled_points,
            colors=colors_final,
            point_size=self.point_size,
            point_shape=self.point_shape,
            visible=visible,
        )
        
        # 记录已加载的点云
        if name not in self._loaded_point_clouds:
            self._loaded_point_clouds.append(name)
        
        CONSOLE.log(f"[green]✓ Added point cloud '{name}' to viewer[/green]")
        CONSOLE.log(f"[cyan]  Total point clouds in viewer: {len(self._loaded_point_clouds)}[/cyan]")
    
    def load_and_display(
        self,
        ply_path: Union[str, Path],
        name: Optional[str] = None,
        visible: bool = True,
    ) -> None:
        """
        加载PLY文件并显示在viewer中
        
        Args:
            ply_path: PLY文件路径
            name: 点云在场景树中的名称，如果为None则使用文件名
            visible: 是否默认可见
        """
        # 加载PLY文件
        points, colors = self.load_ply(ply_path)
        
        # 确定名称
        if name is None:
            name = f"/{Path(ply_path).stem}"
        
        # 添加点云
        self.add_point_cloud(points, colors, name=name, visible=visible)
    
    def load_multiple_ply(
        self,
        ply_paths: List[Union[str, Path]],
        base_name: str = "/point_clouds",
    ) -> None:
        """
        加载多个PLY文件并显示在viewer中
        
        Args:
            ply_paths: PLY文件路径列表
            base_name: 点云在场景树中的基础名称
        """
        for i, ply_path in enumerate(ply_paths):
            name = f"{base_name}/{Path(ply_path).stem}"
            self.load_and_display(ply_path, name=name)
    
    def wait(self) -> None:
        """等待用户中断（保持viewer运行）"""
        try:
            import time
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            CONSOLE.log("\n[yellow]Stopping viewer...[/yellow]")
            self.stop()
    
    def stop(self) -> None:
        """停止viewer服务器"""
        if self.viser_server is not None:
            # viser服务器会在程序退出时自动关闭
            CONSOLE.log("[green]Viewer stopped[/green]")
            self.viser_server = None
            self._loaded_point_clouds.clear()
    
    def get_loaded_point_clouds(self) -> List[str]:
        """获取已加载的点云名称列表"""
        return self._loaded_point_clouds.copy()
    
    def clear_all_point_clouds(self) -> None:
        """清除所有已加载的点云（注意：这不会从viewer中删除，只是清除记录）"""
        self._loaded_point_clouds.clear()
        CONSOLE.log("[yellow]Cleared point cloud records (point clouds still in viewer)[/yellow]")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PLY文件查看器 - 使用nerfstudio viewer在web浏览器中显示点云")
    parser.add_argument(
        "ply_path",
        type=str,
        nargs="+",
        help="PLY文件路径（可以指定多个文件）",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Web服务器主机地址（默认: 0.0.0.0）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Web服务器端口（默认: 自动选择）",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.01,
        help="点云中每个点的大小（默认: 0.01）",
    )
    parser.add_argument(
        "--point-shape",
        type=str,
        default="circle",
        choices=["circle", "square"],
        help="点的形状（默认: circle）",
    )
    
    args = parser.parse_args()
    
    # 创建viewer
    viewer = PLYViewer(
        host=args.host,
        port=args.port,
        point_size=args.point_size,
        point_shape=args.point_shape,
    )
    
    # 启动viewer
    viewer.start_viewer()
    
    # 加载并显示PLY文件
    if len(args.ply_path) == 1:
        viewer.load_and_display(args.ply_path[0])
    else:
        viewer.load_multiple_ply(args.ply_path)
    
    # 保持运行
    viewer.wait()


if __name__ == "__main__":
    main()

