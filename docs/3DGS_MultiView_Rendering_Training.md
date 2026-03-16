# 3DGS 多视角同时渲染与训练（基于 gsplat）

本文档基于 `third_party/gsplat/gsplat` 说明：**在 3D Gaussian Splatting 中是否支持多视角同时渲染与训练**，以及如何正确使用。

---

## 结论摘要

- **可以**：gsplat 的 `rasterization()` 原生支持**一次前向**对 **C 个视角**同时渲染，并返回形状为 `[..., C, height, width, channels]` 的渲染结果。
- **训练**：同一组高斯参数对 C 个视角做一次前向 → 对 C 张图算损失（如 L1 + SSIM）→ 一次 `backward()`，梯度会正确回传到高斯参数，无需循环多视角。

---

## 1. API 支持：Batch Rasterization

### 1.1 接口说明

`gsplat.rendering.rasterization()` 的文档说明（`rendering.py`）：

> **Batch Rasterization**: This function allows for rasterizing a set of 3D Gaussians to a batch of images in one go, by simply providing the batched `viewmats` and `Ks`.

即：通过把多台相机的 `viewmats` 和 `Ks` 在“相机维”上堆叠，即可在一次调用中渲染多张图。

### 1.2 形状约定

| 参数       | 形状说明           | 含义 |
|------------|--------------------|------|
| `viewmats` | `[..., C, 4, 4]`   | C 个相机的世界到相机的变换矩阵 |
| `Ks`       | `[..., C, 3, 3]`   | C 个相机的内参矩阵 |
| `width`    | `int`              | 图像宽度（**所有视角共用**） |
| `height`   | `int`              | 图像高度（**所有视角共用**） |

**返回**：

- `render_colors`: `[..., C, height, width, X]`，X 为通道数（如 RGB=3，RGB+D=4）
- `render_alphas`: `[..., C, height, width, 1]`
- `meta`: 光栅化中间结果（含 `camera_ids`、`batch_ids` 等）

因此，**多视角 = 在传入时把 C 个相机塞进 `viewmats`/`Ks` 的 C 维，一次调用得到 C 张图**。

### 1.3 约束

- **分辨率**：当前接口只接受一个 `width`、一个 `height`，即**所有 C 个视角必须同一分辨率**。若需不同分辨率，需多次调用或自行在外部做缩放/裁剪。
- **颜色**：  
  - 共享颜色：`colors` 为 `[..., N, D]` 或 `[..., N, K, 3]`（SH）。  
  - 每视角不同颜色（如 per-image embedding）：`colors` 可为 `[..., C, N, D]` 或 `[..., C, N, K, 3]`（见 `rendering.py` 中对 `colors` 的 assert）。

---

## 2. 实现层面：多视角如何被处理

- **投影**：`fully_fused_projection` 等接收 `viewmats [..., C, 4, 4]`、`Ks [..., C, 3, 3]`，对每个 (batch, camera) 做 3D→2D 投影，得到 packed 或 un-packed 的 means2d、conics、depths 等，其中相机维 C 被保留或编码在 `camera_ids` 等 meta 中。
- **排序与光栅化**：`isect_offsets` 等形状含 `[..., C, tile_height, tile_width]`，即按 (batch, camera, tile) 组织；CUDA 侧 `rasterize_to_pixels_3dgs_fwd` 等根据 `tile_offsets` 的 image_dims（含 C）一次输出多张图。
- **分布式**：多 GPU 时，文档要求各 rank 的**相机数 C 一致**，高斯可切分到不同 rank；渲染时会在 rank 间交换相机信息，最终每个 rank 得到自己负责的高斯对**全部 C 个视角**的贡献，再聚合得到 C 张图。

因此，**多视角是在同一次前向里、在 C 维上并行完成的**，而不是在 Python 里循环 C 次。

---

## 3. 训练：多视角同时参与一次 step

### 3.1 典型流程

1. **数据**：每个 batch 取 C 个视角（C 个 `camtoworld`、C 个 `K`、C 张 GT 图像），且这 C 张图分辨率相同。
2. **前向**：  
   - `viewmats = torch.linalg.inv(camtoworlds)`，形状 `[B, C, 4, 4]`（若有 batch 维 B）。  
   - `Ks` 形状 `[B, C, 3, 3]`。  
   - 一次调用 `rasterization(..., viewmats=viewmats, Ks=Ks, width=W, height=H)`。  
   - 得到 `render_colors` 形状 `[B, C, H, W, X]`。
3. **损失**：  
   - 将 `render_colors` 与 GT 图像在 `(B, C, H, W)` 上对齐，例如逐像素 L1 + SSIM（或对每视角分别算再平均/求和）。  
   - 损失对 `render_colors` 可微，`render_colors` 对高斯参数可微。
4. **反向**：一次 `loss.backward()`，梯度会从 C 张图一起回传到同一组高斯参数（means、quats、scales、opacities、colors/SH 等）。

这样就是**多视角同时渲染、同时参与训练**，无需对每个视角单独前向或单独 backward。

### 3.2 与 gsplat 官方示例的一致性

`third_party/gsplat/examples/simple_trainer.py` 中：

- 使用 DataLoader 的 `batch_size`（如 4），则 `camtoworlds`、`Ks`、`pixels` 的 batch 维即为相机数（每步 4 个视角）。
- `rasterize_splats()` 内调用 `rasterization(..., viewmats=torch.linalg.inv(camtoworlds), Ks=Ks, ...)`，即一次前向渲染 batch 里的所有视角。
- 损失为 `F.l1_loss(colors, pixels)`（以及 SSIM 等），在整批像素上算，等价于多视角一起参与梯度。

因此，**增大 DataLoader 的 batch_size 即增大每步的视角数 C**，用的就是同一套“多视角同时渲染 + 一次 backward”的机制。

### 3.3 可选：每视角损失权重

若要对不同视角赋权（例如主视角权重大、辅助视角权重小），可在算 loss 时对 `(B, C, H, W)` 按 C 维加权再 reduce，而不是改 gsplat 接口。

---

## 4. 测试与参考

- **单元测试**：`third_party/gsplat/tests/test_rasterization.py` 中已对多视角（如 `C=3`）、多种 `batch_dims`、多种 `render_mode` 做参数化测试，可作回归参考。
- **实现细节**：见 `docs/gsplat_3dgs_summary.md` 中的算法流程与数据结构；多视角对应其中的 `C` 维与 `meta["n_cameras"]` 等。
- **API 与分布式**：见 `gsplat/rendering.py` 中 `rasterization()` 的 docstring（Batch Rasterization、Multi-GPU Distributed Rasterization 两节）。

---

## 5. 小结

| 问题                         | 结论 |
|------------------------------|------|
| 能否多视角同时渲染？         | **能**。传入 `viewmats [..., C, 4, 4]`、`Ks [..., C, 3, 3]`，一次调用得到 `[..., C, H, W, X]`。 |
| 能否多视角同时训练？         | **能**。一次前向渲染 C 张图 → 对 C 张图算 loss → 一次 backward，梯度回到同一组高斯。 |
| 是否要循环每个视角？         | **不需要**。多视角在 C 维上批处理完成。 |
| 所有视角分辨率是否必须相同？ | **是**。当前接口只接受一个 `width`、一个 `height`。 |
| 每视角不同颜色/外观？        | **支持**。可用 `colors` 的 `[..., C, N, D]` 或 `[..., C, N, K, 3]` 形式（见代码 assert）。 |

因此，在 gsplat 下，3DGS 的**多视角同时渲染与训练**是原生支持、可直接使用的；只需组织好 `viewmats`/`Ks` 的 C 维与 DataLoader 的 batch，并保证同一步内的 C 个视角分辨率一致即可。
