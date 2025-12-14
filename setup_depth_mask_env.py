#!/usr/bin/env python3
"""
初始化脚本：检查和下载深度图生成所需的权重文件
运行此脚本以确保所有必需的模型权重都已下载
"""
import os
import sys
import subprocess
from pathlib import Path

def check_conda_env():
    """检查 metric3d conda 环境是否存在"""
    conda_envs = [
        "/root/miniconda3/envs/metric3d",
        "/root/anaconda3/envs/metric3d"
    ]
    
    for env_path in conda_envs:
        python_path = Path(env_path) / "bin" / "python"
        if python_path.exists():
            print(f"✓ 找到 metric3d conda 环境: {env_path}")
            return env_path
    
    print("✗ 未找到 metric3d conda 环境")
    print("  请先创建环境: conda create -n metric3d python=3.8")
    return None

def check_and_download_models():
    """检查并下载所需的模型权重"""
    workspace_root = Path(__file__).parent
    preprocess_dir = workspace_root / "third_party" / "EVolSplat" / "preprocess"
    
    # Metric3D 模型路径
    metric3d_model_dir = preprocess_dir / "metric3d" / "models"
    metric3d_model_path = metric3d_model_dir / "metric_depth_vit_giant2_800k.pth"
    
    print("=" * 60)
    print("检查深度图生成所需的模型权重")
    print("=" * 60)
    
    # 检查 conda 环境
    print("\n检查 conda 环境...")
    env_path = check_conda_env()
    if not env_path:
        print("  警告: metric3d 环境不存在，但可以继续检查模型")
    else:
        # 检查关键依赖
        print("\n检查 metric3d 环境依赖...")
        python_path = Path(env_path) / "bin" / "python"
        try:
            import subprocess
            result = subprocess.run(
                [str(python_path), "-c", "import mmcv; print('OK')"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print("  ✓ mmcv 已安装")
            else:
                print("  ✗ mmcv 未正确安装")
                print(f"    请运行: bash {workspace_root}/install_metric3d_deps.sh")
        except Exception as e:
            print(f"  ⚠ 无法检查依赖: {e}")
    
    # 检查 Metric3D 模型
    print(f"\n检查 Metric3D 模型...")
    print(f"  路径: {metric3d_model_path}")
    
    if metric3d_model_path.exists():
        size_mb = metric3d_model_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ 模型已存在 ({size_mb:.1f} MB)")
        if size_mb < 100:
            print(f"  ⚠ 警告: 模型文件可能不完整（大小异常小）")
    else:
        print(f"  ✗ 模型不存在，需要下载")
        print(f"  正在下载 Metric3D 模型（约 813 MB）...")
        
        # 确保目录存在
        metric3d_model_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用 download_models.sh 脚本下载
        download_script = preprocess_dir / "download_models.sh"
        if download_script.exists():
            try:
                print(f"  执行下载脚本: {download_script}")
                result = subprocess.run(
                    ["bash", str(download_script)],
                    cwd=str(preprocess_dir),
                    capture_output=False,  # 显示实时输出
                    timeout=3600  # 60分钟超时
                )
                if result.returncode == 0:
                    if metric3d_model_path.exists():
                        size_mb = metric3d_model_path.stat().st_size / (1024 * 1024)
                        print(f"  ✓ 模型下载完成 ({size_mb:.1f} MB)")
                    else:
                        print(f"  ⚠ 下载脚本执行成功，但模型文件未找到")
                        return False
                else:
                    print(f"  ✗ 下载失败，返回码: {result.returncode}")
                    print(f"  请手动运行: bash {download_script}")
                    return False
            except subprocess.TimeoutExpired:
                print(f"  ✗ 下载超时（超过60分钟）")
                print(f"  请手动运行: bash {download_script}")
                return False
            except Exception as e:
                print(f"  ✗ 下载出错: {e}")
                print(f"  请手动运行: bash {download_script}")
                return False
        else:
            print(f"  ✗ 下载脚本不存在: {download_script}")
            print(f"  请手动下载模型到: {metric3d_model_path}")
            print(f"  下载链接: https://drive.google.com/file/d/1KVINiBkVpJylx_6z1lAC7CQ4kmn-RJRN/view?usp=drive_link")
            return False
    
    # 检查 Metric3D 路径
    metric3d_path = preprocess_dir / "metric3d"
    if not metric3d_path.exists():
        print(f"\n✗ Metric3D 代码目录不存在: {metric3d_path}")
        return False
    
    test_script = metric3d_path / "mono" / "tools" / "test_scale_cano.py"
    if not test_script.exists():
        print(f"\n✗ Metric3D 测试脚本不存在: {test_script}")
        return False
    
    config_file = metric3d_path / "mono" / "configs" / "HourglassDecoder" / "vit.raft5.giant2.py"
    if not config_file.exists():
        print(f"\n✗ Metric3D 配置文件不存在: {config_file}")
        return False
    
    print(f"\n✓ Metric3D 环境检查通过")
    
    # 检查 OneFormer 脚本
    print(f"\n检查 OneFormer 脚本...")
    oneformer_script = preprocess_dir / "gen_semantic_oneformer.py"
    if oneformer_script.exists():
        print(f"  ✓ OneFormer 脚本存在: {oneformer_script}")
        print(f"  注意: OneFormer 模型会自动从 Hugging Face 下载（首次运行时）")
    else:
        print(f"  ⚠ OneFormer 脚本不存在: {oneformer_script}")
        print(f"  语义分割功能可能无法使用")
    
    # 打印环境变量设置
    print("\n" + "=" * 60)
    print("环境变量设置（已自动配置在 launch.json 中）:")
    print("=" * 60)
    print(f"  METRIC3D_PATH={metric3d_path}")
    print(f"  METRIC3D_MODEL_PATH={metric3d_model_path}")
    print(f"\n注意: OneFormer 使用 Hugging Face Transformers，模型会自动下载")
    print(f"  模型名称: shi-labs/oneformer_cityscapes_swin_large")
    print(f"  下载位置: ~/.cache/huggingface/hub/")
    
    print("\n" + "=" * 60)
    print("✓ 所有检查完成！现在可以在 VS Code 中使用 launch.json 运行脚本")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = check_and_download_models()
    sys.exit(0 if success else 1)
