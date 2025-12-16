# -*- coding: utf-8 -*-
"""
从 processed 结构读取 Waymo 数据，在任意相机位姿下生成 LiDAR RGB 投影图（含“动态物体优先”渲染），
并严格对齐你给出的 waymo_preprocess.py / waymo_sourceloader.py 的坐标与数据约定。

关键对齐点：
- ego_pose/{t}.txt            : 保存的是 frame.pose.transform (Vehicle->World) 的 4x4 绝对位姿。
- extrinsics/{cam}.txt        : 保存的是 camera.extrinsic.transform (Camera->Vehicle)，为 Waymo 相机坐标。
                                为与 OpenCV 投影一致，先右乘 OPENCV2DATASET 将相机坐标改为 OpenCV，相机外参再取逆得到 Vehicle->Camera(OpenCV)。
- intrinsics/{cam}.txt        : 保存 1D 向量 [fx, fy, cx, cy, k1, k2, p1, p2, k3]；投影仅用前 4 项，按缩放比例调整。
- lidar/{t}.bin               : 每行至少 13/14 个 float32，列结构与你的 loader 一致；我们取第 3~5 列为点 (x,y,z)，位于车辆坐标系。
- instances/instances_info.json + frame_instances.json:
                                使用你保存的“物体到世界”位姿 (o2w) 与尺寸，按帧筛出实例，检查点是否在框内；
                                局部坐标用 p_local = (o2w^-1 * p_world)[:3]，并缓存到 intid2inboxpoints。

输出：
- {project_root}/frame_points.pkl : (frame_points, waymoid2intid, intid2inboxpoints) —— 与你原版一致的三元组！
- {project_root}/cond/*.png       : LiDAR 彩色投影（虚拟位姿）
- {project_root}/lq/*.png         : 对齐分辨率的对应相机图

运行示例见文件尾 main()。
"""

import os
import cv2
import json
import math
import shutil
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# OPENCV2DATASET = np.array([[0, -1, 0, 0],
#                            [0,  0,-1, 0],
#                            [1,  0, 0, 0],
#                            [0,  0, 0, 1]], dtype=np.float32)


def inv_se3(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def filter_duplicates(uv, dists, colors):
    """与原版一致：同像素(u,v)只保留距离最近的点（Z-buffer）"""
    data = np.column_stack((uv, dists, colors))
    df = pd.DataFrame(data, columns=['u','v','dist','r','g','b'])
    df_sorted = df.sort_values(by='dist', ascending=True).groupby(['u','v'], as_index=False).first()
    uv_f = df_sorted[['u','v']].values.astype(np.int32)
    d_f  = df_sorted['dist'].values.reshape(-1,1).astype(np.float32)
    col  = df_sorted[['r','g','b']].values.astype(np.uint8)
    return uv_f, d_f, col

def load_intrinsics(path_txt):
    """
    读取 intrinsics/{cam}.txt: 1D 数组 [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    返回 3x3 K；k/p 不参与投影（保持与你现有管线一致）。
    """
    arr = np.loadtxt(path_txt).astype(np.float32).reshape(-1)
    assert arr.size >= 4, f"Invalid intrinsics at {path_txt}"
    fx, fy, cx, cy = arr[0], arr[1], arr[2], arr[3]
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,   0,  1]], dtype=np.float32)
    return K

def get_opencv2dataset_matrix(dataset: str) -> np.ndarray:
    """
    返回 OpenCV(右-下-前) -> 数据集相机坐标 的 4x4 齐次变换矩阵。
    - waymo:   OpenCV -> WaymoCam
    - kitti:   OpenCV == KITTI 相机坐标，返回 I
    - nuscenes:通常存的是 OpenCV 相机外参，也返回 I（如你在 preprocess 中另有定义，再按需改）
    """
    if dataset.lower() == "waymo":
        return np.array(
            [[0, 0, 1, 0],
             [-1, 0, 0, 0],
             [0, -1, 0, 0],
             [0, 0, 0, 1]], dtype=np.float32)
    elif dataset.lower() in ("kitti", "nuscenes", "argoverse"):
        return np.eye(4, dtype=np.float32)
    else:
        # 兜底：维持之前的行为（Waymo）
        return np.eye(4, dtype=np.float32)

def load_extrinsics_vehicle_to_camera(path_txt, dataset="waymo"):
    # 盘里存的是：Camera(dataset) -> Vehicle
    T_v_from_c_ds = np.loadtxt(path_txt).astype(np.float32)   # Cam(ds)->Veh

    # 让它接收 OpenCV 相机坐标的点：右乘 OpenCV->Dataset
    T_v_from_c_cv = T_v_from_c_ds @ get_opencv2dataset_matrix(dataset)

    # 我们投影更喜欢 Vehicle->Camera(OpenCV)
    T_c_from_v_cv = np.linalg.inv(T_v_from_c_cv)
    return T_c_from_v_cv

def load_ego_vehicle_to_world(path_txt):
    """
    读取 ego_pose/{t}.txt: 保存 frame.pose.transform，即 Vehicle->World 的 4x4 绝对位姿
    """
    T_vw = np.loadtxt(path_txt).astype(np.float32)  # Vehicle->World
    assert T_vw.shape == (4,4), f"Bad ego pose: {path_txt}"
    return T_vw

def load_lidar_points_vehicle(path_bin, dataset="waymo"):
    """
    读取 lidar/{t}.bin：
      你的 preprocess 写入了 [origins(3), points(3), flows(3), ground(1), intensity(1), elongation(1), laser_id(1)]
      共 13 列；你的 loader 以 -1x14 读取（包含 flow_class），我们这里兼容两种：13 或 14 列。
    返回：
      pts_v: (N,3) —— 车辆坐标系下的点（取列 3:6）
    """
    buf = np.fromfile(path_bin, dtype=np.float32) 
    if dataset == "waymo":
        pts = buf.reshape(-1, 14)[:, 3:6]
    elif dataset in ("kitti", "argoverse", "nuscenes"):
        pts = buf.reshape(-1, 4)[:, :3]
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    pts_v = pts.astype(np.float32)
    return pts_v

_INSTANCE_CACHE = {}

def _load_instances_json(scene_dir):
    key = os.path.abspath(scene_dir)
    if key in _INSTANCE_CACHE:
        return _INSTANCE_CACHE[key]

    info_path  = os.path.join(scene_dir, "instances", "instances_info.json")
    frame_path = os.path.join(scene_dir, "instances", "frame_instances.json")
    if not (os.path.exists(info_path) and os.path.exists(frame_path)):
        # 若缺失实例文件，返回空
        _INSTANCE_CACHE[key] = ({}, {}, {})
        return _INSTANCE_CACHE[key]

    with open(info_path, "r") as f:
        instances_info = json.load(f)  # keys: "0","1",...
    with open(frame_path, "r") as f:
        frame_instances = json.load(f) # keys: "0","1",... -> [ids]

    # 将 instances_info 预处理为：每个 id -> {frame_idx: (T_ow, size)}
    id2framePoseSize = {}
    for sid_str, rec in instances_info.items():
        sid = int(sid_str)
        frames = rec["frame_annotations"]["frame_idx"]
        poses  = rec["frame_annotations"]["obj_to_world"]
        sizes  = rec["frame_annotations"]["box_size"]
        mapping = {}
        for fi, pose, sz in zip(frames, poses, sizes):
            T_ow = np.array(pose, dtype=np.float32).reshape(4,4)  # Object->World
            sz   = np.array(sz, dtype=np.float32).reshape(3,)     # [l,w,h]
            mapping[int(fi)] = (T_ow, sz)
        id2framePoseSize[sid] = mapping

    # 构建稳定的 int id（1..M），保持与“旧代码的 waymoid2intid”风格一致
    all_ids = sorted([int(k) for k in instances_info.keys()])
    waymoid2intid = {sid: i+1 for i, sid in enumerate(all_ids)}  # 外部可用：原始（简化）id -> 连续 int

    _INSTANCE_CACHE[key] = (waymoid2intid, id2framePoseSize, frame_instances)
    return _INSTANCE_CACHE[key]

def get_instances_for_frame(scene_dir, time_id):
    """
    返回当前帧的实例列表，每项：(intid, T_ow(4x4), size(3,))
    """
    waymoid2intid, id2framePoseSize, frame_instances = _load_instances_json(scene_dir)
    out = []
    if not frame_instances:
        return waymoid2intid, out
    key = str(time_id)
    if key not in frame_instances:
        return waymoid2intid, out
    for sid in frame_instances[key]:
        # sid 已是简化 id（int）
        sid = int(sid)
        if sid in id2framePoseSize and time_id in id2framePoseSize[sid]:
            T_ow, sz = id2framePoseSize[sid][time_id]
            intid = waymoid2intid[sid]
            out.append((intid, T_ow, sz))
    return waymoid2intid, out

# ---------- 投影/着色 ----------

def project_points_to_image(points_w, T_cw, K, img_size):
    W, H = img_size
    pts_h = np.concatenate([points_w, np.ones((points_w.shape[0],1), np.float32)], 1)
    pc = (T_cw @ pts_h.T).T[:, :3]    # World->Camera(OpenCV)
    z = pc[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return (np.zeros((0,2), np.int32), np.zeros((0,1), np.float32), np.zeros((0,), np.int64))
    pc = pc[valid]
    uv = (K @ pc.T).T
    uv = uv[:, :2] / pc[:, 2:3]
    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)
    in_img = (u>=0)&(u<W)&(v>=0)&(v<H)
    u, v = u[in_img], v[in_img]
    d = np.linalg.norm(pc[in_img], axis=1, keepdims=True).astype(np.float32)
    if uv.shape[0] == 0:
        # 没有任何点进来，打印前几个点投影前后的 z
        # 这段只在调试时打开，避免频繁 I/O
        pc_dbg = (T_cw @ np.concatenate([points_w[:8], np.ones((min(8, points_w.shape[0]),1), np.float32)], 1).T).T
        zpos = (pc_dbg[:,2] > 0).sum()
        # print(f"[debug] uv=0, pc_z>0: {zpos}/{pc_dbg.shape[0]}")
    return np.stack([u, v], 1), d, np.where(valid)[0][in_img]

def colorize_points_vehicle(scene_dir, timestep, pts_v, resomult=0.5, dataset="waymo"):
    """
    用该帧多相机图给车辆坐标系点上色；并输出其世界坐标副本。
    返回：
      pts_vrgb: (N,6) 车辆坐标 + RGB(float)
      pts_wrgb: (N,6) 世界坐标 + RGB(float)
    """
    T_vw = load_ego_vehicle_to_world(os.path.join(scene_dir, "lidar_pose" if dataset == "nuscenes" else "ego_pose", f"{timestep:03d}.txt"))
    pts_w = (T_vw[:3,:3] @ pts_v.T + T_vw[:3,3:4]).T

    rgb = np.zeros((pts_v.shape[0],3), dtype=np.uint8)

    if dataset == "nuscenes":
        CAMERA_PRIORITY = [0, 1, 2, 3, 4, 5]
    elif dataset == "argoverse":
        CAMERA_PRIORITY = [0, 5, 6, 1, 2, 3, 4]
    elif dataset == "waymo":
        CAMERA_PRIORITY = [0, 1, 2, 3, 4]
    elif dataset == "kitti":
        CAMERA_PRIORITY = [0, 1]
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    for cam_id in CAMERA_PRIORITY:
        img_path = os.path.join(scene_dir, "images", f"{timestep:03d}_{cam_id}.jpg")
        K_path   = os.path.join(scene_dir, "intrinsics", f"{cam_id}.txt")
        # E_path   = os.path.join(scene_dir, "extrinsics", f"{cam_id}.txt")
        if not (os.path.exists(img_path) and os.path.exists(K_path)):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        H0, W0 = img.shape[:2]
        W = int(round(W0 * resomult)); H = int(round(H0 * resomult))
        if W <= 0 or H <= 0:
            continue

        K = load_intrinsics(K_path).copy()
        K[0,0]*=resomult; K[1,1]*=resomult; K[0,2]*=resomult; K[1,2]*=resomult

        if dataset == "nuscenes":
            # nuScenes: 外参是 Cam(dataset)->World，且逐帧
            E_path_frame = os.path.join(scene_dir, "extrinsics", f"{timestep:03d}_{cam_id}.txt")
            if not os.path.exists(E_path_frame):
                continue
            T_w_from_c_ds = np.loadtxt(E_path_frame).astype(np.float32)  # Cam(ds) -> World
            # 将相机坐标从 dataset 相机系“右乘”到 OpenCV 相机系（nuScenes 下为 I，不变）
            T_w_from_c_cv = T_w_from_c_ds @ get_opencv2dataset_matrix(dataset)
            T_cw = np.linalg.inv(T_w_from_c_cv)  # World -> Cam(OpenCV)

        else:
            # Waymo/KITTI: 外参是 Cam(dataset)->Ego（基本不随时间）
            E_path = os.path.join(scene_dir, "extrinsics", f"{cam_id}.txt")
            if not os.path.exists(E_path):
                continue
            T_c_from_v = load_extrinsics_vehicle_to_camera(E_path, dataset=dataset)  # Veh -> Cam(OpenCV)
            T_cw = T_c_from_v @ np.linalg.inv(T_vw)  # World -> Cam(OpenCV)
        
        uv, d, idx = project_points_to_image(pts_w, T_cw, K, (W, H))
        if uv.shape[0] == 0:
            continue

        img_small = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        bgr = img_small[uv[:,1], uv[:,0]]
        rgb[idx] = bgr[:, ::-1]  # BGR->RGB 覆盖

    pts_vrgb = np.concatenate([pts_v, rgb.astype(np.float32)], axis=1)
    pts_wrgb = np.concatenate([pts_w, rgb.astype(np.float32)], axis=1)
    return pts_vrgb, pts_wrgb

def build_frame_points_and_objects(scene_dir, resomult=0.5, dataset="waymo"):
    """
    与原版功能/命名一致，返回：
      frame_points: list[lenT]，每项 (Nb,6) 世界坐标背景点 + RGB
      waymoid2intid: dict 映射（简化实例 id -> 连续 int，从 1 开始）
      intid2inboxpoints: dict[intid] -> { frame_idx -> (N,6) 局部坐标 + RGB }
         （局部坐标定义：p_local = (o2w^-1 * p_world)[:3]）
    """
    lidar_dir = os.path.join(scene_dir, "lidar")
    timesteps = sorted([int(fn.split('.')[0]) for fn in os.listdir(lidar_dir) if fn.endswith(".bin")])
    if not timesteps:
        raise FileNotFoundError(f"No lidar frames under {lidar_dir}")

    # 从 JSON 预加载全部实例映射
    waymoid2intid_global, _, _ = _load_instances_json(scene_dir)
    # 但 intid2inboxpoints 只在出现点时填入
    intid2inboxpoints = {}
    frame_points = []

    for i, t in enumerate(tqdm(timesteps, desc=f"[{os.path.basename(scene_dir)}] build frame_points & objects")):
        pts_v = load_lidar_points_vehicle(os.path.join(lidar_dir, f"{t:03d}.bin"), dataset=dataset)

        # 给点上色，同时拿到世界坐标
        pts_vrgb, pts_wrgb = colorize_points_vehicle(scene_dir, t, pts_v, resomult=resomult, dataset=dataset)

        # 取当前帧的实例（intid, T_ow, size）
        waymoid2intid, inst_list = get_instances_for_frame(scene_dir, i)
        # 若未保存 instances/，退化为没有动态物体
        if not waymoid2intid_global:
            waymoid2intid_global = waymoid2intid

        any_obj_mask = np.zeros((pts_wrgb.shape[0],), dtype=bool)

        # 将每个实例的 in-box 点转为“局部坐标 + RGB”，缓存到 intid2inboxpoints[intid][i]
        for (intid, T_ow, size_lwh) in inst_list:
            # World->Object
            T_wo = inv_se3(T_ow)

            # 计算每个点在物体局部的坐标
            pw = pts_wrgb[:, :3]
            pw_h = np.concatenate([pw, np.ones((pw.shape[0],1), dtype=np.float32)], axis=1)
            po = (T_wo @ pw_h.T).T[:, :3]  # (N,3)

            half = size_lwh.astype(np.float32) / 2.0
            m = (np.abs(po) <= (half + 1e-6)).all(axis=1)  # in-box mask
            if not np.any(m):
                continue

            po_rgb = np.concatenate([po[m], pts_wrgb[m, 3:]], axis=1).astype(np.float32)
            any_obj_mask |= m

            if intid not in intid2inboxpoints:
                intid2inboxpoints[intid] = {}
            intid2inboxpoints[intid][i] = po_rgb

        # 背景点：不属于任何实例的点，直接保存世界坐标 + RGB
        bg = pts_wrgb[~any_obj_mask]
        frame_points.append(bg.astype(np.float32))

    # waymoid2intid：若 instances 缺失，返回空映射；否则用全局映射
    waymoid2intid_out = waymoid2intid_global if waymoid2intid_global else {}

    return frame_points, waymoid2intid_out, intid2inboxpoints

# ---------- 多帧融合 ----------

def merge_multiframe_points(frame_points, idx, multiframe_num=4):
    n = len(frame_points)
    if idx < multiframe_num:
        l, r = 0, min(2*multiframe_num+1, n)
    elif idx > n-1-multiframe_num:
        l, r = max(0, n-(2*multiframe_num+1)), n
    else:
        l, r = idx-multiframe_num, idx+multiframe_num+1
    return np.concatenate(frame_points[l:r], axis=0)

# ---------- 渲染（先动态，后背景） ----------

def render_with_dynamics(scene_dir,
                         frame_points,
                         waymoid2intid,
                         intid2inboxpoints,
                         frame_idx,
                         time_id,
                         T_cw_virtual,
                         cam_id,
                         resomult=0.5,
                         bg_spec=None,
                         dyn_spec=None,
                         radius=2):
    """
    bg_spec: FrameSpec，决定合并哪些背景帧来渲染
    dyn_spec: FrameSpec，决定为每个实例取哪些帧的局部点来渲染
    """
    # 读取分辨率和 K
    img_path = os.path.join(scene_dir, "images", f"{frame_idx:03d}_{cam_id}.jpg")
    K_path   = os.path.join(scene_dir, "intrinsics", f"{cam_id}.txt")
    if not (os.path.exists(img_path) and os.path.exists(K_path)):
        raise FileNotFoundError(f"Missing image/intrinsics for cam {cam_id}, frame {frame_idx:03d}")
    img0 = cv2.imread(img_path, cv2.IMREAD_COLOR)
    H0, W0 = img0.shape[:2]
    W = int(round(W0 * resomult)); H = int(round(H0 * resomult))
    K = load_intrinsics(K_path).copy()
    K[0,0]*=resomult; K[1,1]*=resomult; K[0,2]*=resomult; K[1,2]*=resomult

    img  = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.ones((H, W, 3), dtype=np.uint8)

    n_total = len(frame_points)
    bg_spec = bg_spec or FrameSpec("pm", K=4, S=1)
    dyn_spec = dyn_spec or FrameSpec("pm", K=1, S=1)

    # 背景帧选择并合并
    bg_indices = bg_spec.select(n_total, frame_idx)
    merged_bg = np.concatenate([frame_points[i] for i in bg_indices], axis=0) if len(bg_indices) > 0 else np.zeros((0,6), np.float32)

    # 当前帧实例
    _, inst_list = get_instances_for_frame(scene_dir, time_id)

    # 收集实例局部点（按 dyn_spec 指定的帧集合）
    objs_world_rgb, objs_depth = [], []
    dyn_indices = dyn_spec.select(n_total, frame_idx)

    for (intid, T_ow, size_lwh) in inst_list:
        # 汇总该实例在 dyn_indices 里出现的局部点
        buf = []
        if intid in intid2inboxpoints:
            for fi in dyn_indices:
                if fi in intid2inboxpoints[intid]:
                    buf.append(intid2inboxpoints[intid][fi])
        if not buf:
            continue
        pinbox = np.concatenate(buf, axis=0)  # (K,6) [x_local, y_local, z_local, r,g,b]

        # 排序依据：局部 xy 的最小半径
        min_r = np.linalg.norm(pinbox[:, :2], axis=1).min() if pinbox.shape[0] > 0 else 1e9
        objs_depth.append(min_r)

        # 局部点 -> 世界（利用绝对 T_ow）
        R_o = T_ow[:3, :3]; t_o = T_ow[:3, 3]
        pw  = (R_o @ pinbox[:, :3].T).T + t_o[None, :]
        rgb = pinbox[:, 3:].astype(np.float32)
        objs_world_rgb.append(np.concatenate([pw, rgb], axis=1).astype(np.float32))

    # 近 -> 远排序
    if objs_world_rgb:
        order = np.argsort(np.asarray(objs_depth))
        objs_world_rgb = [objs_world_rgb[i] for i in order]

    # 先画动态
    for arr in objs_world_rgb:
        uv, d, idx = project_points_to_image(arr[:, :3], T_cw_virtual, K, (W, H))
        if uv.shape[0] == 0:
            continue
        uv_f, _, col = filter_duplicates(uv, d, arr[idx, 3:])
        temp = np.zeros_like(img)
        temp[uv_f[:,1], uv_f[:,0]] = col.astype(np.uint8)
        img += temp * mask
        if uv_f.shape[0] > 0 and radius > 0:
            added = [uv_f + np.array([dx, dy]) for dx in range(-radius, radius+1) for dy in range(-radius, radius+1)]
            added = np.concatenate([uv_f] + added, axis=0)
            valid = (added[:,0]>=0)&(added[:,0]<W)&(added[:,1]>=0)&(added[:,1]<H)
            added = added[valid]
            mask[added[:,1], added[:,0]] = 0

    # 再画背景
    if merged_bg.shape[0] > 0:
        uv, d, idx = project_points_to_image(merged_bg[:, :3], T_cw_virtual, K, (W, H))
        if uv.shape[0] > 0:
            uv_f, _, col = filter_duplicates(uv, d, merged_bg[idx, 3:])
            temp = np.zeros_like(img)
            temp[uv_f[:,1], uv_f[:,0]] = col.astype(np.uint8)
            img += temp * mask

    return img, (W, H)

# ---------- 与原版接口一致的批量投影 ----------

def project_images_from_processed(scene_dir,
                                  txt_file_list,
                                  output_path,
                                  resomult=0.5,
                                  default_bg="pm4x1",
                                  default_dyn="pm1x1",
                                  dataset="waymo"):
    os.makedirs(os.path.join(output_path, "cond"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "lq"),  exist_ok=True)

    cache_pkl = os.path.join(output_path, "frame_points.pkl")
    if os.path.exists(cache_pkl):
        with open(cache_pkl, "rb") as f:
            frame_points, waymoid2intid, intid2inboxpoints = pickle.load(f)
    else:
        frame_points, waymoid2intid, intid2inboxpoints = build_frame_points_and_objects(scene_dir, resomult=resomult, dataset=dataset)
        with open(cache_pkl, "wb") as f:
            # 与原版完全一致：只存三元组
            pickle.dump((frame_points, waymoid2intid, intid2inboxpoints), f)

    for fpath in tqdm(txt_file_list, desc="Rendering (with dynamics)"):
        name, frame_idx, cam_id, time_id, bg_spec, dyn_spec = parse_filename_for_meta(fpath, default_bg, default_dyn)

        # 1) 读 T_rel（cam_rel：相机->世界'）
        T_rel = np.loadtxt(fpath).astype(np.float32)

        # 2) cam_rel -> 绝对世界 -> T_cw
        first_pose = load_ego_vehicle_to_world(os.path.join(scene_dir, "lidar_pose" if dataset == "nuscenes" else "ego_pose", "000.txt"))
        T_w_from_c_abs = first_pose @ T_rel
        T_cw_virtual   = np.linalg.inv(T_w_from_c_abs)

        # 3) 渲染（把 bg_spec / dyn_spec 传进去）
        img, (W, H) = render_with_dynamics(
            scene_dir=scene_dir,
            frame_points=frame_points,
            waymoid2intid=waymoid2intid,
            intid2inboxpoints=intid2inboxpoints,
            frame_idx=frame_idx,
            time_id=time_id,
            T_cw_virtual=T_cw_virtual,
            cam_id=cam_id,
            resomult=resomult,
            bg_spec=bg_spec,
            dyn_spec=dyn_spec,
            radius=2,
        )
        img_file = fpath.replace(".txt", ".png")
        img_file_name = os.path.basename(img_file)
        cv2.imwrite(os.path.join(output_path, "cond", img_file_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


        # 生成/拷贝 lq（保证与 cond 同分辨率）
        if os.path.exists(img_file):
            src = cv2.imread(img_file, cv2.IMREAD_COLOR)
            if src is not None:
                lq = cv2.resize(src, (W, H), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(output_path, "lq", img_file_name), lq)
            # else:
            #     shutil.copy(img_file, os.path.join(output_path, "lq", img_file))
        # else:
        #     raw_img_path = os.path.join(scene_dir, "images", f"{frame_id:03d}_{cam_id}.jpg")
        #     if os.path.exists(raw_img_path):
        #         src = cv2.imread(raw_img_path, cv2.IMREAD_COLOR)
        #         lq = cv2.resize(src, (W, H), interpolation=cv2.INTER_LINEAR)
        #         cv2.imwrite(os.path.join(output_path, "lq", img_file), lq)

class FrameSpec:
    def __init__(self, kind, **kw):
        self.kind = kind   # "pm", "cur", "list", "abs"
        self.kw = kw       # {"K":int,"S":int} / {"indices":list} / {"start":int,"end":int,"step":int}

    def select(self, n_frames, center_idx):
        if self.kind == "cur":
            return np.array([np.clip(center_idx, 0, n_frames-1)], dtype=int)

        if self.kind == "pm":
            K = int(self.kw.get("K", 0))
            S = int(self.kw.get("S", 1))
            rel = np.arange(-K, K+1, S, dtype=int)
            idx = center_idx + rel
            return idx[(idx >= 0) & (idx < n_frames)]

        if self.kind == "list":
            idx = np.array(self.kw.get("indices", []), dtype=int)
            return idx[(idx >= 0) & (idx < n_frames)]

        if self.kind == "abs":
            start = int(self.kw.get("start", 0))
            end   = int(self.kw.get("end", 0))
            step  = int(self.kw.get("step", 1))
            if step == 0:
                step = 1
            if end < start:
                start, end = end, start
            idx = np.arange(start, end+1, step, dtype=int)
            return idx[(idx >= 0) & (idx < n_frames)]

        # fallback
        return np.array([np.clip(center_idx, 0, n_frames-1)], dtype=int)
    
def parse_framespec(spec_str):
    """把字符串规格解析为 FrameSpec；None 或空则返回默认 pm4x1。"""
    if spec_str is None:
        return FrameSpec("pm", K=4, S=1)

    s = spec_str.strip().lower()

    if s == "cur":
        return FrameSpec("cur")

    if s.startswith("pm"):
        # pm{K}x{S} ；步长可选，默认为1
        body = s[2:]
        if "x" in body:
            K, S = body.split("x", 1)
            return FrameSpec("pm", K=int(K), S=int(S))
        else:
            return FrameSpec("pm", K=int(body), S=1)

    if s.startswith("list-"):
        items = s[5:]
        idxs = [int(x) for x in items.split(",") if x]
        return FrameSpec("list", indices=idxs)

    if s.startswith("abs-"):
        body = s[4:]
        # 形如 start..end:step 或 start..end
        if ":" in body:
            rng, step = body.split(":", 1)
            step = int(step)
        else:
            rng, step = body, 1
        if ".." in rng:
            start, end = rng.split("..", 1)
            return FrameSpec("abs", start=int(start), end=int(end), step=step)

    # 兜底：默认 pm4x1
    return FrameSpec("pm", K=4, S=1)

def parse_filename_for_meta(filename, default_bg="pm4x1", default_dyn="pm1x1"):
    """
    解析 {name}_{frame_idx}_{cam_id}_{time_id}[_bg-XXX][_dyn-YYY].txt
    返回：name, frame_idx, cam_id, bg_spec(FrameSpec), dyn_spec(FrameSpec)
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    if len(parts) < 2:
        # 兜底：从内容推不出就默认 0
        raise ValueError(f"Invalid filename: {filename}")
    else:
        try:
            name = parts[0]
            frame_idx = int(parts[1])
            cam_id = int(parts[2])
            time_id = int(parts[3])
        except:
            raise ValueError(f"Invalid filename: {filename}")

    bg_str, dyn_str = default_bg, default_dyn
    for p in parts[4:]:
        q = p.lower()
        if q.startswith("bg-"):
            bg_str = q[3:]
        elif q.startswith("dyn-"):
            dyn_str = q[4:]

    return name, frame_idx, cam_id, time_id, parse_framespec(bg_str), parse_framespec(dyn_str)



# ---------- CLI：保持你旧版 main 的使用习惯 ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", type=str, required=True,
                        help="ProjectPath/data/waymo/processed/training")
    parser.add_argument("--scene_id", type=str, required=True,
                        help="如 001、002 ...")
    parser.add_argument("--project_root", type=str, required=True,
                        help="输出目录 & 缓存 frame_points.pkl 所在目录；也扫描这里的子目录里的 *.txt/*.png")
    parser.add_argument("--resomult", type=float, default=0.5)
    parser.add_argument("--default_bg", type=str, default="pm5x1")
    parser.add_argument("--default_dyn", type=str, default="pm10x1")
    parser.add_argument("--dataset", type=str, default="waymo")
    args = parser.parse_args()

    scene_dir = os.path.join(args.processed_root, args.scene_id)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"Scene dir not found: {scene_dir}")

    txt_file_list = []
    project_path = os.path.join(args.project_root, args.scene_id)
    subdir = os.path.join(project_path, "sub")
    if not os.path.isdir(subdir):
        os.makedirs(subdir, exist_ok=True)
        txt_file_list = []
    else:
        for fn in os.listdir(subdir):
            if not fn.endswith(".txt"):
                continue
            fpath = os.path.join(subdir, fn)
            txt_file_list.append(fpath)

    #txt_file_list = txt_file_list[::16]

    project_images_from_processed(
        scene_dir=scene_dir,
        txt_file_list=txt_file_list,
        output_path=project_path,
        resomult=args.resomult,
        default_bg=args.default_bg,
        default_dyn=args.default_dyn,
        dataset=args.dataset
    )
    print("All done.")


if __name__ == "__main__":
    main()