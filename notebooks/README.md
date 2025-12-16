# Jupyter Notebook 使用说明

## metric3d 环境配置

Jupyter Notebook 已成功配置在 conda 环境 `metric3d` 中。

## 启动 Jupyter Notebook

### 方法1: 在 metric3d 环境中启动

```bash
# 激活环境
conda activate metric3d

# 启动 Jupyter Notebook
cd /root/drivestudio-coding/notebooks
jupyter notebook
```

### 方法2: 指定端口启动（适用于远程服务器）

```bash
conda activate metric3d
cd /root/drivestudio-coding/notebooks
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## 选择 Kernel

在 Jupyter Notebook 中打开 notebook 后：

1. 点击右上角的 "Kernel" 菜单
2. 选择 "Change Kernel"
3. 选择 **"Python (metric3d)"** 作为 kernel

或者，在创建新 notebook 时，直接选择 "Python (metric3d)" kernel。

## 已安装的包

- Jupyter Notebook 7.3.3
- JupyterLab 4.3.8
- ipykernel 6.29.5
- IPython 8.12.3

## 可用的 Kernels

运行以下命令查看所有可用的 kernels：

```bash
conda activate metric3d
jupyter kernelspec list
```

当前可用的 kernels：
- `python3`: 默认 Python kernel
- `metric3d`: metric3d 环境的 kernel

## 使用 nuscenes_pcd_generation.ipynb

1. 启动 Jupyter Notebook（使用上述方法）
2. 打开 `nuscenes_pcd_generation.ipynb`
3. 确保选择了 "Python (metric3d)" kernel
4. 按顺序执行所有单元格

## 注意事项

- 确保在 metric3d 环境中启动 Jupyter Notebook
- 如果遇到导入错误，检查是否选择了正确的 kernel
- 所有依赖包应该在 metric3d 环境中已安装

