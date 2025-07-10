# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python
# brisque_demo.py
"""
计算一张图片的 BRISQUE 分数（piq 实现）
用法: python brisque_demo.py <image_path>
"""

import sys
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as F
import piq

def brisque_score(img_path: Path, device: str = "cpu") -> float:
    """返回 BRISQUE 分数；值越低表示画质越好"""
    img = Image.open(img_path).convert("RGB")
    tensor = F.to_tensor(img).unsqueeze(0).to(device)  # [1,3,H,W] 0‑1
    with torch.no_grad():
        score = piq.brisque(tensor, data_range=1.0).item()
    return score

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python brisque_demo.py <image_path>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"找不到文件: {path}")
        sys.exit(1)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"BRISQUE score: {brisque_score(path, dev):.4f}  (越低越好)")
