# Stage4.0 说明：为什么 `_compute_2d_features_for_gaussians` 之前没实现？

这份说明用于回答一个常见疑问：  
“`@models/streetforward/minimal_trainer_stage3_3.py` 里不是已经有 2D 特征流程了吗，为什么 Stage4.0 还会报 `_compute_2d_features_for_gaussians` 不存在？”

---

## 结论先说

- `stage3_3` **确实有** 2D 特征流程；
- 但它使用的是 **`_compute_2d_features_bg_distant(...)`**（面向 bg+distant 的成对接口）；
- 代码里并不存在一个通用的 **`_compute_2d_features_for_gaussians(...)`** 方法；
- Stage4.0 新增 rigid 分支时，调用了这个“通用命名”的方法，但继承链里没人实现，所以触发 `AttributeError`。

---

## 继承链和方法来源

`MinimalStreetForwardStage3_3` 的继承是：

- `MinimalStreetForwardStage3_3`
  -> `MinimalStreetForwardStage3_2`
  -> `MinimalStreetForwardStage3_1`
  -> `MinimalStreetForwardStage3_2d`

在这条链里，已有的 2D 相关核心方法是：

- `_compute_2d_features_bg_distant(...)`（定义在 `minimal_trainer_stage3_2d.py`）

它的设计目标是“给 **bg+distant** 两个分支一起算特征并按索引切分返回”，不是“对任意 gaussians 集合做通用 2D 特征提取”。

所以：

- 你在 `stage3_3.py` 能看到 `_compute_2d_features_bg_distant` 的调用；
- 但找不到 `_compute_2d_features_for_gaussians` 的定义，这是正常的（之前从未需要该通用接口）。

---

## 为什么 Stage4.0 会暴露这个问题

Stage4.0 需要给 rigid 分支单独走一条 2D-only 路径，输入是“仅 rigid 的 gaussians 子集”。  
这和 `bg+distant` 的成对接口不完全匹配，于是实现里自然会倾向写一个通用 helper：

- 输入：任意 gaussians + source views/images
- 输出：该集合对应的 2D backproject 特征

如果这个 helper 没在类里实现，就会出现你看到的报错：

- `AttributeError: 'MinimalStreetForwardStage4_0' object has no attribute '_compute_2d_features_for_gaussians'`

---

## 修复后的状态

Stage4.0 已补上 `_compute_2d_features_for_gaussians(...)`，内部复用了和 Stage3 2D 同源的流程：

1. `render_rgb_only`
2. 构造 6 通道输入（原图 + 渲染图）
3. `image_feature_extractor`
4. `render_and_backproject_streaming`

这样 rigid 子集就能独立拿到 2D 特征，不再依赖“只能 bg+distant 配对”接口。

---

## 经验总结（后续避免同类问题）

- 看到“有 2D 流程”不等于“有通用 helper”；
- 新分支（如 rigid）接入时，优先确认：
  - 现有方法是“任务特化接口”还是“可复用通用接口”；
  - 方法名是否只在某个类出现调用、但未在父类定义；
- 对这类 helper，建议在类定义时显式声明，避免运行时才暴露缺失。

