#!/usr/bin/env python3
# Copyright 2025-2026 Abdolvahab Khalili Sadaghiani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""YOLO (frozen) -> GNN bounding-box refiner.

The detections of a frozen Ultralytics YOLO model become the nodes of a
weighted graph (IoU and k-nearest-neighbour edges).  A small graph neural
network predicts an additive correction (dx1, dy1, dx2, dy2) for every box and
is trained with a Smooth-L1 loss against YOLO-format ground truth.  Only the
GNN, its feature MLP and the class embedding are trained.
"""

import os
import sys
import json
import math
import time
import random
import pickle
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Ultralytics YOLO and PyTorch Geometric -----------------
# Not installed at run time: on Ubuntu 24.04 / ROS 2 Jazzy the system Python is
# externally managed and a silent "pip install" fails.  See README.md.
try:
    from ultralytics import YOLO
    from torch_geometric.nn import GCNConv, GraphConv
except ImportError as exc:
    raise ImportError(
        f"Missing Python package '{exc.name or exc}'. Install it in the Python "
        "environment that runs ROS 2, e.g. 'pip install ultralytics torch_geometric' "
        "(see README.md, Installation).") from exc


# =============================== Utilities ====================================

def set_seed(seed: int = 2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def list_images(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png")
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)])


def compute_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter + 1e-6
    return float(inter / denom) if denom > 0 else 0.0


def greedy_match(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray, iou_thresh: float = 0.5) -> List[Tuple[int,int]]:
    """Greedy one-to-one matching by IoU >= threshold."""
    if pred_xyxy.size == 0 or gt_xyxy.size == 0:
        return []
    P, G = pred_xyxy.shape[0], gt_xyxy.shape[0]
    iou_mat = np.zeros((P, G), dtype=np.float32)
    for i in range(P):
        for j in range(G):
            iou_mat[i, j] = compute_iou_xyxy(pred_xyxy[i], gt_xyxy[j])

    matches, used_p, used_g = [], set(), set()
    while True:
        i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        best = iou_mat[i, j]
        if best < iou_thresh:
            break
        if i in used_p or j in used_g:
            iou_mat[i, j] = -1.0
            continue
        matches.append((i, j))
        used_p.add(i); used_g.add(j)
        iou_mat[i, :] = -1.0
        iou_mat[:, j] = -1.0
    return matches


def load_yolo_txt(txt_path: str, W: int, H: int) -> List[List[float]]:
    """
    YOLO label .txt format: 'cls cx cy w h' (normalized).
    Returns a list of [x1,y1,x2,y2] in pixels.
    """
    out = []
    if not os.path.exists(txt_path):
        return out
    with open(txt_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            _, cx, cy, w, h = p[:5]
            cx, cy, w, h = map(float, (cx, cy, w, h))
            x1 = (cx - w / 2.0) * W; y1 = (cy - h / 2.0) * H
            x2 = (cx + w / 2.0) * W; y2 = (cy + h / 2.0) * H
            out.append([x1, y1, x2, y2])
    return out


def clip_boxes_xyxy(boxes_np: np.ndarray, W: int, H: int) -> np.ndarray:
    boxes_np[:, 0] = np.clip(boxes_np[:, 0], 0, W - 1)
    boxes_np[:, 1] = np.clip(boxes_np[:, 1], 0, H - 1)
    boxes_np[:, 2] = np.clip(boxes_np[:, 2], 0, W - 1)
    boxes_np[:, 3] = np.clip(boxes_np[:, 3], 0, H - 1)
    return boxes_np


def strip_ros_args(argv: List[str]) -> List[str]:
    """Remove the '--ros-args ... [--]' blocks that ROS 2 launch appends."""
    out, in_ros = [], False
    for a in argv:
        if a == "--ros-args":
            in_ros = True
        elif in_ros and a == "--":
            in_ros = False
        elif not in_ros:
            out.append(a)
    return out


def resolve_model_path(model: str) -> str:
    """
    Find the YOLO weights given by --model.
    Order: the path as given; <share>/object_detector_husky/weights/<name> of the
    installed package; weights/<name> of a source checkout.  Anything not found
    is returned unchanged (Ultralytics downloads official names such as yolov8n.pt).
    """
    if os.path.isfile(model):
        return model
    name = os.path.basename(model)
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.join(
            get_package_share_directory("object_detector_husky"), "weights", name))
    except Exception:  # not in a ROS 2 environment, or the package is not installed
        pass
    src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(src_root, "weights", name))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return model


def class_name(names, idx: int) -> str:
    """Class name from a YOLO 'names' dict or list; 'C<idx>' if unknown."""
    if isinstance(names, dict) and idx in names:
        return str(names[idx])
    if isinstance(names, (list, tuple)) and 0 <= idx < len(names):
        return str(names[idx])
    return f"C{idx}"


# ========================= Graph construction (weighted) =======================

def centers_wh_from_boxes(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = np.maximum(1.0, x2 - x1)
    h = np.maximum(1.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return cx, cy, w, h


def build_weighted_graph(boxes_t: torch.Tensor, H: int, W: int,
                         iou_thr: float = 0.25, knn: int = 5, alpha_iou: float = 0.7
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted edge: w_ij = alpha*IoU + (1-alpha)*exp(-dist/sigma), with symmetric edges.
    Also adds kNN neighbors by center distance to ensure connectivity.
    Returns: (edge_index [2,E], edge_weight [E]).
    """
    bx = boxes_t.detach().cpu().numpy()
    N = bx.shape[0]
    if N <= 1:
        ei = torch.zeros((2, 0), dtype=torch.long)
        ew = torch.zeros((0,), dtype=torch.float32)
        return ei, ew

    cx, cy, w, h = centers_wh_from_boxes(bx)
    max_hw = max(H, W)
    sigma = max(10.0, 0.10 * max_hw)

    edges = []
    ew = []
    # Precompute distances and IoUs
    for i in range(N):
        row_dists = []
        for j in range(N):
            if i == j:
                continue
            dx = cx[i] - cx[j]; dy = cy[i] - cy[j]
            dist = math.sqrt(dx * dx + dy * dy)
            iou = compute_iou_xyxy(bx[i], bx[j])
            wij = alpha_iou * iou + (1.0 - alpha_iou) * math.exp(-dist / sigma)

            # IoU-based connection
            if iou >= iou_thr:
                edges.append([i, j]); edges.append([j, i])
                ew.append(wij); ew.append(wij)

            row_dists.append((dist, j, wij))

        # kNN connection (by center distance)
        row_dists.sort(key=lambda t: t[0])
        for k in range(min(knn, len(row_dists))):
            _, j, wij = row_dists[k]
            edges.append([i, j]); edges.append([j, i])
            ew.append(wij); ew.append(wij)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(ew, dtype=torch.float32)
    return edge_index, edge_weight


# =========================== Node features & model =============================

class FeatureMLP(nn.Module):
    """
    Maps raw node features (geometry + score + class embedding) -> in_channels.
    raw_dim = 7 + emb_dim
    """
    def __init__(self, emb_dim: int = 16, in_channels: int = 128):
        super().__init__()
        self.emb_dim = emb_dim
        self.in_channels = in_channels
        raw_dim = 7 + emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU()
        )

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.mlp(raw)


class GNNRefiner(nn.Module):
    """
    Lightweight GNN for bbox refinement.
    - conv_type: "gcn" | "graph"  (GCNConv or GraphConv)
    - L graph layers with dropout and optional residual.
    - Outputs per-node deltas [dx1, dy1, dx2, dy2] (additive XYXY).
    """
    def __init__(self,
                 num_classes: int = 80,
                 conv_type: str = "gcn",
                 layers: int = 3,
                 in_channels: int = 128,
                 hidden_channels: int = 128,
                 out_channels: int = 128,
                 emb_dim: int = 16,
                 dropout: float = 0.10,
                 use_residual: bool = True):
        super().__init__()
        assert layers >= 2, "Use at least 2 GNN layers for stable refinement."
        self.num_classes = num_classes
        self.conv_type = conv_type.lower()
        self.layers = layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_residual = use_residual

        # Trainable class embedding
        self.lbl_emb = nn.Embedding(num_classes, emb_dim)
        # Feature projector
        self.feat_mlp = FeatureMLP(emb_dim=emb_dim, in_channels=in_channels)

        # Build GNN layers
        self.gnn = nn.ModuleList()
        prev = in_channels
        for li in range(layers):
            nxt = out_channels if li == (layers - 1) else hidden_channels
            if self.conv_type == "gcn":
                self.gnn.append(GCNConv(prev, nxt))
            elif self.conv_type == "graph":
                self.gnn.append(GraphConv(prev, nxt))
            else:
                raise ValueError("conv_type must be 'gcn' or 'graph'")
            prev = nxt

        # Output head: regress 4 deltas
        self.head = nn.Linear(out_channels, 4)

    def _make_node_features(self, boxes: torch.Tensor, scores: torch.Tensor,
                            labels: torch.Tensor, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        boxes: [N,4] XYXY (float32), scores: [N] (float32), labels: [N] (long)
        Returns: x [N, in_channels]
        """
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w = torch.clamp(x2 - x1, min=1.0)
        h = torch.clamp(y2 - y1, min=1.0)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        cx_n = cx / float(W)
        cy_n = cy / float(H)
        w_n = w / float(W)
        h_n = h / float(H)
        area = w_n * h_n
        aspect = w / (h + 1e-6)

        geom = torch.stack([cx_n, cy_n, w_n, h_n, area, aspect, scores.to(device)], dim=1)
        emb = self.lbl_emb(labels.to(device))
        raw = torch.cat([geom, emb], dim=1)
        x = self.feat_mlp(raw)
        return x

    def forward(self, boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor,
                H: int, W: int, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Returns per-node deltas [N,4].
        """
        device = boxes.device
        x = self._make_node_features(boxes, scores, labels, H, W, device)
        for i, layer in enumerate(self.gnn):
            if isinstance(layer, (GCNConv, GraphConv)):
                h = layer(x, edge_index, edge_weight=edge_weight)
            else:
                h = layer(x, edge_index)
            h = F.relu(h)
            if self.use_residual and h.shape == x.shape:
                h = h + x
            if self.dropout > 0 and i < len(self.gnn) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
            x = h
        deltas = self.head(x)
        return deltas


# ============================= Caching YOLO outputs ============================

def run_yolo_and_cache(yolo: YOLO, img_paths: List[str], device: str,
                       cache_path: Optional[str], cap_nodes: int,
                       conf_thr_keep: float = 0.0) -> Dict[str, dict]:
    """
    Runs YOLO on img_paths and caches results as:
      cache[img_path] = {
        "H": int, "W": int,
        "boxes": torch.FloatTensor [N,4] (XYXY, on CPU),
        "scores": torch.FloatTensor [N],
        "labels": torch.LongTensor [N]
      }
    Keeps top-N by score; filters scores < conf_thr_keep if >0.
    cache_path=None keeps the cache in memory only (nothing read or written).
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        # Sanity: filter out images not in current list
        wanted = set(img_paths)
        cache = {k: v for k, v in cache.items() if k in wanted}
        print(f"[CACHE] Loaded {len(cache)} entries from {cache_path}")
    else:
        cache = {}

    to_do = [p for p in img_paths if p not in cache]
    if len(to_do) == 0:
        return cache

    print(f"[YOLO] Running YOLO for {len(to_do)} images to populate cache...")
    for idx, p in enumerate(to_do, 1):
        im = cv2.imread(p)
        if im is None:
            continue
        H, W = im.shape[:2]
        res = yolo(p, device=device, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu()
        scores = res.boxes.conf.cpu()
        labels = res.boxes.cls.cpu().long()

        if conf_thr_keep > 0.0:
            keep = scores >= conf_thr_keep
            boxes = boxes[keep]; scores = scores[keep]; labels = labels[keep]

        # keep top-N by confidence
        if boxes.size(0) > 0:
            order = torch.argsort(scores, descending=True)
            boxes = boxes[order]; scores = scores[order]; labels = labels[order]
            if boxes.size(0) > cap_nodes:
                boxes = boxes[:cap_nodes]; scores = scores[:cap_nodes]; labels = labels[:cap_nodes]

        cache[p] = {"H": H, "W": W, "boxes": boxes, "scores": scores, "labels": labels}

        # Progress print occasionally
        if idx % 100 == 0 or idx == len(to_do):
            print(f"  cached {idx}/{len(to_do)}")

    # Save cache
    if cache_path:
        safe_makedirs(os.path.dirname(cache_path))
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"[CACHE] Saved YOLO cache => {cache_path} (size: {len(cache)})")
    return cache


# ================================ Training/Eval ================================

def smooth_l1_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(a, b, beta=1.0, reduction="mean")


def apply_deltas_xyxy(boxes: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    out = boxes.clone()
    out[:, 0] += deltas[:, 0]
    out[:, 1] += deltas[:, 1]
    out[:, 2] += deltas[:, 2]
    out[:, 3] += deltas[:, 3]
    return out


def train_one_epoch(refiner: GNNRefiner, optimizer, yolo_cache: Dict[str, dict],
                    annot_dir: str, device: str,
                    iou_thr_graph: float, knn: int, alpha_iou: float,
                    score_weighting: bool = False) -> float:
    refiner.train()
    loss_sum = 0.0
    steps = 0

    keys = list(yolo_cache.keys())
    random.shuffle(keys)

    for img_path in keys:
        entry = yolo_cache[img_path]
        H, W = entry["H"], entry["W"]
        boxes = entry["boxes"]
        scores = entry["scores"]
        labels = entry["labels"]

        if boxes.size(0) == 0:  # no proposals
            continue

        # Load GT labels
        base = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = os.path.join(annot_dir, base + ".txt")
        gt_list = load_yolo_txt(gt_path, W, H)
        if len(gt_list) == 0:
            continue
        gt = torch.tensor(gt_list, dtype=torch.float32, device=device)

        # Build weighted graph (on CPU -> to device)
        ei, ew = build_weighted_graph(boxes, H, W, iou_thr_graph, knn, alpha_iou)
        ei = ei.to(device); ew = ew.to(device)

        # Move proposals to device
        boxes_d = boxes.to(device)
        scores_d = scores.to(device)
        labels_d = labels.to(device)

        # Forward GNN
        deltas = refiner(boxes_d, scores_d, labels_d, H, W, ei, ew)
        refined = apply_deltas_xyxy(boxes_d, deltas)

        # Greedy match (use numpy for IoU; both on CPU)
        pairs = greedy_match(refined.detach().cpu().numpy(), gt.detach().cpu().numpy(), iou_thresh=0.5)
        if len(pairs) == 0:
            continue

        N = refined.size(0)
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        target = refined.detach().clone()
        for pi, gi in pairs:
            mask[pi] = True
            target[pi] = gt[gi]

        pred_l = refined[mask]
        targ_l = target[mask]
        loss = smooth_l1_loss(pred_l, targ_l)

        if score_weighting:
            # Reweight loss per-node by YOLO confidence (average)
            sw = scores_d[mask]
            if sw.numel() > 0:
                loss = (loss * sw.mean())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item())
        steps += 1

    return loss_sum / max(1, steps)


@torch.no_grad()
def evaluate(refiner: GNNRefiner, yolo_cache: Dict[str, dict], annot_dir: str, device: str,
             iou_thr_graph: float, knn: int, alpha_iou: float) -> Tuple[float, float]:
    """
    Returns: (val_loss, mean IoU gain) over matched pairs.
    mean IoU gain = mean( IoU(refined, gt) - IoU(original, gt) ) across all matched pairs.
    """
    refiner.eval()
    loss_sum = 0.0
    steps = 0

    iou_gains = []
    for img_path, entry in yolo_cache.items():
        H, W = entry["H"], entry["W"]
        boxes = entry["boxes"]; scores = entry["scores"]; labels = entry["labels"]
        if boxes.size(0) == 0:
            continue

        base = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = os.path.join(annot_dir, base + ".txt")
        gt_list = load_yolo_txt(gt_path, W, H)
        if len(gt_list) == 0:
            continue
        gt = torch.tensor(gt_list, dtype=torch.float32, device=device)

        ei, ew = build_weighted_graph(boxes, H, W, iou_thr_graph, knn, alpha_iou)
        ei = ei.to(device); ew = ew.to(device)
        boxes_d = boxes.to(device); scores_d = scores.to(device); labels_d = labels.to(device)

        deltas = refiner(boxes_d, scores_d, labels_d, H, W, ei, ew)
        refined = apply_deltas_xyxy(boxes_d, deltas)

        # Compute loss on matched pairs (as in train)
        pairs = greedy_match(refined.detach().cpu().numpy(), gt.detach().cpu().numpy(), iou_thresh=0.5)
        if len(pairs) == 0:
            continue

        N = refined.size(0)
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        target = refined.detach().clone()
        for pi, gi in pairs:
            mask[pi] = True
            target[pi] = gt[gi]

        pred_l = refined[mask]
        targ_l = target[mask]
        loss = smooth_l1_loss(pred_l, targ_l)
        loss_sum += float(loss.item())
        steps += 1

        # IoU gain (per match) using numpy for clarity
        ref_np = refined.detach().cpu().numpy()
        orig_np = boxes.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        for pi, gi in pairs:
            iou_ref = compute_iou_xyxy(ref_np[pi], gt_np[gi])
            iou_ori = compute_iou_xyxy(orig_np[pi], gt_np[gi])
            iou_gains.append(iou_ref - iou_ori)

    val_loss = loss_sum / max(1, steps)
    mean_iou_gain = float(np.mean(iou_gains)) if len(iou_gains) > 0 else 0.0
    return val_loss, mean_iou_gain


# ================================= Detection ==================================

@torch.no_grad()
def detect_and_save(refiner: GNNRefiner, yolo: YOLO, img_dir: str, out_dir: str, device: str,
                    cap_nodes: int, score_thr_draw: float,
                    iou_thr_graph: float, knn: int, alpha_iou: float, names=None):
    safe_makedirs(out_dir)
    img_paths = list_images(img_dir)
    print(f"[DETECT] {len(img_paths)} images in {img_dir}")

    for p in img_paths:
        im = cv2.imread(p)
        if im is None:
            continue
        H, W = im.shape[:2]
        res = yolo(p, device=device, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu()
        scores = res.boxes.conf.cpu()
        labels = res.boxes.cls.cpu().long()

        if boxes.size(0) == 0:
            cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), im)
            continue

        # keep top-N
        idx = torch.argsort(scores, descending=True)
        boxes = boxes[idx]; scores = scores[idx]; labels = labels[idx]
        if boxes.size(0) > cap_nodes:
            boxes = boxes[:cap_nodes]; scores = scores[:cap_nodes]; labels = labels[:cap_nodes]

        ei, ew = build_weighted_graph(boxes, H, W, iou_thr_graph, knn, alpha_iou)
        ei = ei.to(device); ew = ew.to(device)
        boxes_d = boxes.to(device); scores_d = scores.to(device); labels_d = labels.to(device)

        deltas = refiner(boxes_d, scores_d, labels_d, H, W, ei, ew)
        refined = apply_deltas_xyxy(boxes_d, deltas).detach().cpu().numpy().astype(int)

        # draw refined
        out = im.copy()
        for (x1, y1, x2, y2), sc, lb in zip(refined, scores.numpy(), labels.numpy()):
            if sc < score_thr_draw:
                continue
            x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2)); y2 = max(0, min(H - 1, y2))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"{class_name(names, int(lb))}:{sc:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), out)


# ================================== Main ======================================

def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        prog="object_detector_husky",
        description="YOLO (frozen) -> GNN bounding-box refiner; only the GNN is trained.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Modes
    parser.add_argument("--mode", type=str, default="train_detect",
                        choices=["train", "detect", "train_detect"],
                        help="train the refiner, detect with it, or both")
    # Paths
    parser.add_argument("--model", type=str, default="weights.pt",
                        help="YOLO weights; a bare file name is also looked up in the "
                             "package's weights/ folder")
    parser.add_argument("--train_dir", type=str,
                        default="/proj/vahabkhalili/users/x_abdkh/coco_data_auto/train2017",
                        help="training images (.jpg, .jpeg, .png)")
    parser.add_argument("--train_annot", type=str,
                        default="/proj/vahabkhalili/users/x_abdkh/coco_data_auto/train2017",
                        help="YOLO .txt labels, one file per training image (same stem)")
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Fraction of train_dir used for validation (random split).")
    parser.add_argument("--test_dir", type=str,
                        default="/proj/vahabkhalili/users/x_abdkh/test_results",
                        help="images to run the refined detector on")
    parser.add_argument("--out_dir", type=str,
                        default="/proj/vahabkhalili/users/x_abdkh/test_results_COCO",
                        help="output folder: config, cache, checkpoint, annotated images")
    parser.add_argument("--gnn_ckpt", type=str, default=None,
                        help="GNN checkpoint written by training and read by detection "
                             "(None: <out_dir>/best_weights.pt)")
    # Training
    parser.add_argument("--epochs", type=int, default=5, help="training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--device", type=str, default="cuda", help="cuda, cuda:N or cpu")
    parser.add_argument("--seed", type=int, default=2024,
                        help="seed of the train/val split, the initialisation and the order")
    parser.add_argument("--score_weighting", action="store_true",
                        help="Weight loss by mean YOLO confidence of matched nodes.")
    # Proposals / graph
    parser.add_argument("--cap_nodes", type=int, default=20,
                        help="keep at most this many detections (graph nodes) per image")
    parser.add_argument("--conf_thr_keep", type=float, default=0.0,
                        help="If >0, drop YOLO proposals with score < thr before top-N.")
    parser.add_argument("--iou_thr_graph", type=float, default=0.25,
                        help="IoU at or above which two detections get an IoU edge")
    parser.add_argument("--knn", type=int, default=5,
                        help="nearest neighbours (by box centre) linked to every node")
    parser.add_argument("--alpha_iou", type=float, default=0.7,
                        help="edge weight = alpha*IoU + (1-alpha)*exp(-distance/sigma)")
    # GNN sizing
    parser.add_argument("--conv_type", type=str, default="gcn", choices=["gcn", "graph"],
                        help="GCNConv or GraphConv layers")
    parser.add_argument("--layers", type=int, default=3, help="number of GNN layers (>= 2)")
    parser.add_argument("--in_channels", type=int, default=128,
                        help="width of the feature MLP output")
    parser.add_argument("--hidden_channels", type=int, default=128,
                        help="width of the hidden GNN layers")
    parser.add_argument("--out_channels", type=int, default=128,
                        help="width of the last GNN layer")
    parser.add_argument("--emb_dim", type=int, default=16, help="size of the class embedding")
    parser.add_argument("--dropout", type=float, default=0.10,
                        help="dropout after every GNN layer but the last")
    parser.add_argument("--use_residual", action="store_true",
                        help="add the layer input to its output when the widths match")
    # Detection drawing
    parser.add_argument("--score_thr_draw", type=float, default=0.5,
                        help="draw refined boxes whose YOLO score is at least this")
    # Caching
    parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=True,
                        help="keep the YOLO outputs of train/val in <out_dir>/cache and "
                             "reuse them; --no-use_cache keeps them in memory only")
    args = parser.parse_args(strip_ros_args(sys.argv[1:] if argv is None else argv))

    set_seed(args.seed)
    safe_makedirs(args.out_dir)
    model_path = resolve_model_path(args.model)

    # Persist config
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(dict(vars(args), model_resolved=model_path), f, indent=2)

    # Load YOLO (frozen)
    print(f"[INFO] Loading => {model_path}")
    yolo = YOLO(model_path)
    yolo.to(args.device)
    yolo.eval()
    for p in yolo.model.parameters():
        p.requires_grad = False
    try:
        names = yolo.names
    except Exception:
        names = {}
    num_classes = len(names) if names else 80

    do_train = args.mode in ("train", "train_detect")
    do_detect = args.mode in ("detect", "train_detect")
    best_ckpt = args.gnn_ckpt or os.path.join(args.out_dir, "best_weights.pt")

    if do_train:
        # Split train/val
        all_imgs = list_images(args.train_dir)
        if len(all_imgs) == 0:
            print(f"[ERROR] No images in train_dir: {args.train_dir}")
            sys.exit(1)

        random.shuffle(all_imgs)
        n_val = max(1, int(len(all_imgs) * args.val_split))
        val_imgs = sorted(all_imgs[:n_val])
        train_imgs = sorted(all_imgs[n_val:])

        print(f"[DATA] Train images: {len(train_imgs)} | Val images: {len(val_imgs)}")

        # YOLO outputs for train/val: on disk and reused unless --no-use_cache.
        # The file names carry everything that changes their content.
        train_cache_path = val_cache_path = None
        if args.use_cache:
            cache_dir = os.path.join(args.out_dir, "cache")
            safe_makedirs(cache_dir)
            tag = (f"{os.path.splitext(os.path.basename(model_path))[0]}"
                   f"_cap{args.cap_nodes}_conf{args.conf_thr_keep:g}")
            train_cache_path = os.path.join(cache_dir, f"yolo_train_{tag}.pkl")
            val_cache_path = os.path.join(cache_dir, f"yolo_val_{tag}.pkl")
        else:
            print("[WARN] --no-use_cache: YOLO runs over train/val on every start.")
        train_cache = run_yolo_and_cache(yolo, train_imgs, args.device, train_cache_path,
                                         cap_nodes=args.cap_nodes,
                                         conf_thr_keep=args.conf_thr_keep)
        val_cache = run_yolo_and_cache(yolo, val_imgs, args.device, val_cache_path,
                                       cap_nodes=args.cap_nodes,
                                       conf_thr_keep=args.conf_thr_keep)

    # Build GNN refiner
    refiner = GNNRefiner(
        num_classes=num_classes,
        conv_type=args.conv_type,
        layers=args.layers,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        use_residual=args.use_residual
    ).to(args.device)

    # Optimizer
    optimizer = torch.optim.Adam(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")

    # ------------------------------- TRAIN -----------------------------------
    if do_train:
        print("[INFO] Start training...")
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss = train_one_epoch(
                refiner, optimizer, train_cache, args.train_annot, args.device,
                iou_thr_graph=args.iou_thr_graph, knn=args.knn, alpha_iou=args.alpha_iou,
                score_weighting=args.score_weighting
            )
            val_loss, mean_iou_gain = evaluate(
                refiner, val_cache, args.train_annot, args.device,
                iou_thr_graph=args.iou_thr_graph, knn=args.knn, alpha_iou=args.alpha_iou
            )
            dt = time.time() - t0
            print(f"[EPOCH {ep}/{args.epochs}] "
                  f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"mean_IoU_gain={mean_iou_gain:+.4f}  time={dt:.1f}s")

            # Save best
            if val_loss < best_val:
                best_val = val_loss
                safe_makedirs(os.path.dirname(os.path.abspath(best_ckpt)))
                torch.save(refiner.state_dict(), best_ckpt)
                print(f"[CKPT] Saved best GNN checkpoint => {best_ckpt}")

    # ------------------------------- DETECT ----------------------------------
    if do_detect:
        # An untrained refiner would add random offsets: require a checkpoint.
        if not os.path.exists(best_ckpt):
            print(f"[ERROR] No GNN checkpoint at {best_ckpt}. "
                  "Train first (--mode train) or pass --gnn_ckpt.")
            sys.exit(1)
        refiner.load_state_dict(torch.load(best_ckpt, map_location=args.device))
        print(f"[CKPT] Loaded best GNN checkpoint => {best_ckpt}")
        det_dir = os.path.join(args.out_dir, "detections_refined")
        print("[INFO] Running refined detection on test images...")
        detect_and_save(
            refiner=refiner,
            yolo=yolo,
            img_dir=args.test_dir,
            out_dir=det_dir,
            device=args.device,
            cap_nodes=args.cap_nodes,
            score_thr_draw=args.score_thr_draw,
            iou_thr_graph=args.iou_thr_graph,
            knn=args.knn,
            alpha_iou=args.alpha_iou,
            names=names
        )
        print(f"[INFO] Wrote refined detections to: {det_dir}")


if __name__ == "__main__":
    main()
