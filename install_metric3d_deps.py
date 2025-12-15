#!/usr/bin/env python3
"""
安装 Metric3D 所需的依赖包
"""
import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """安装 Metric3D 依赖"""
    workspace_root = Path(__file__).parent
    preprocess_dir = workspace_root / "third_party" / "EVolSplat" / "preprocess"
    requirements_file = preprocess_dir / "metric3d" / "requirements.txt"
    
    # 检查 conda 环境
    conda_envs = [
        "/root/miniconda3/envs/metric3d",
        "/root/anaconda3/envs/metric3d"
    ]
    
    python_path = None
    for env_path in conda_envs:
        test_path = Path(env_path) / "bin" / "python"
        if test_path.exists():
            python_path = str(test_path)
            print(f"✓ 找到 metric3d conda 环境: {env_path}")
            break
    
    if not python_path:
        print("✗ 未找到 metric3d conda 环境")
        print("  请先创建环境: conda create -n metric3d python=3.8")
        return False
    
    pip_path = str(Path(python_path).parent / "pip")
    
    print("=" * 60)
    print("安装 Metric3D 依赖包")
    print("=" * 60)
    print(f"使用 Python: {python_path}")
    print()
    
    # 安装基础依赖
    print("1. 安装 requirements.txt 中的依赖...")
    try:
        result = subprocess.run(
            [pip_path, "install", "-q", "-r", str(requirements_file)],
            check=True,
            timeout=600
        )
        print("  ✓ 基础依赖安装完成")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ 安装失败: {e}")
        return False
    except subprocess.TimeoutExpired:
        print("  ✗ 安装超时")
        return False
    
    # 安装 openmim
    print("\n2. 安装 openmim...")
    try:
        subprocess.run(
            [pip_path, "install", "-q", "-U", "openmim"],
            check=True,
            timeout=300
        )
        print("  ✓ openmim 安装完成")
    except Exception as e:
        print(f"  ⚠ openmim 安装失败: {e}")
    
    # 安装 mmengine
    print("\n3. 安装 mmengine...")
    try:
        subprocess.run(
            [python_path, "-m", "mim", "install", "mmengine"],
            check=True,
            timeout=300
        )
        print("  ✓ mmengine 安装完成")
    except Exception as e:
        print(f"  ✗ mmengine 安装失败: {e}")
        return False
    
    # 安装 mmcv-full
    print("\n4. 安装 mmcv-full==1.7.1...")
    try:
        subprocess.run(
            [python_path, "-m", "mim", "install", "mmcv-full==1.7.1"],
            check=True,
            timeout=600
        )
        print("  ✓ mmcv-full 安装完成")
    except Exception as e:
        print(f"  ✗ mmcv-full 安装失败: {e}")
        return False
    
    # 安装 mmsegmentation
    print("\n5. 安装 mmsegmentation==0.30.0...")
    try:
        subprocess.run(
            [pip_path, "install", "-q", "mmsegmentation==0.30.0"],
            check=True,
            timeout=600
        )
        print("  ✓ mmsegmentation 安装完成")
    except Exception as e:
        print(f"  ✗ mmsegmentation 安装失败: {e}")
        return False
    
    # 安装其他依赖
    print("\n6. 安装其他依赖...")
    try:
        subprocess.run(
            [pip_path, "install", "-q", "numpy==1.20.0", "scikit-image==0.18.0"],
            check=True,
            timeout=300
        )
        print("  ✓ 其他依赖安装完成")
    except Exception as e:
        print(f"  ⚠ 其他依赖安装失败: {e}")
    
    # 安装 OneFormer 依赖（用于语义分割）
    print("\n7. 安装 OneFormer 依赖（transformers等）...")
    try:
        subprocess.run(
            [pip_path, "install", "-q", "transformers", "torch", "pillow", "opencv-python-headless", "tqdm"],
            check=True,
            timeout=600
        )
        print("  ✓ OneFormer 依赖安装完成")
    except Exception as e:
        print(f"  ⚠ OneFormer 依赖安装失败: {e}")
        print("  注意: 如果语义分割功能需要，请手动安装: pip install transformers")
    
    # 验证安装
    print("\n" + "=" * 60)
    print("验证安装...")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [python_path, "-c", "import mmcv; print('✓ mmcv version:', mmcv.__version__)"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print("  ✗ mmcv 验证失败")
    except Exception as e:
        print(f"  ⚠ mmcv 验证出错: {e}")
    
    try:
        result = subprocess.run(
            [python_path, "-c", "import mmengine; print('✓ mmengine installed')"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("  ✓ mmengine 验证通过")
        else:
            print("  ✗ mmengine 验证失败")
    except Exception as e:
        print(f"  ⚠ mmengine 验证出错: {e}")
    
    try:
        result = subprocess.run(
            [python_path, "-c", "import mmsegmentation; print('✓ mmsegmentation installed')"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("  ✓ mmsegmentation 验证通过")
        else:
            print("  ✗ mmsegmentation 验证失败")
    except Exception as e:
        print(f"  ⚠ mmsegmentation 验证出错: {e}")
    
    print("\n" + "=" * 60)
    print("✓ 依赖安装完成！")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = install_dependencies()
    sys.exit(0 if success else 1)

