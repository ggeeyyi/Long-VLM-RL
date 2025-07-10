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
"""
测试 WanProcessor 的功能
"""
import torch
from verl.utils.tokenizer import get_processor

def test_wan_processor():
    """测试 WanProcessor 的基本功能"""
    print("测试 WanProcessor...")
    
    # 模拟一个 wan 模型路径
    # 注意：这里需要替换为实际的 wan 模型路径
    model_path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # 请替换为实际路径
    
    try:
        # 获取 processor
        processor = get_processor(model_path, diffusion=True, device_map="auto")
        
        if processor is not None:
            print(f"✅ 成功创建 WanProcessor: {type(processor)}")
            
            # 测试编码功能
            test_prompt = "A beautiful sunset over the mountains"
            prompt_embeds, negative_prompt_embeds = processor(
                prompt=test_prompt,
                negative_prompt="low quality, blurry",
                device="cuda"
            )
            
            print(f"✅ Prompt 编码成功:")
            print(f"   - Prompt embeddings shape: {prompt_embeds.shape}")
            print(f"   - Negative prompt embeddings shape: {negative_prompt_embeds.shape}")
            
            # 测试批量处理
            batch_prompts = ["A cat playing", "A dog running"]
            batch_embeds, batch_negative_embeds = processor(
                prompt=batch_prompts,
                negative_prompt="bad quality",
                device="cuda"
            )
            
            print(f"✅ 批量处理成功:")
            print(f"   - Batch embeddings shape: {batch_embeds.shape}")
            print(f"   - Batch negative embeddings shape: {batch_negative_embeds.shape}")
            
        else:
            print("❌ Processor 为 None")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请确保模型路径正确并且模型文件存在")

def test_processor_attributes():
    """测试 WanProcessor 的属性"""
    print("\n测试 WanProcessor 属性...")
    
    try:
        from verl.utils.wan_processor import WanProcessor
        from transformers import AutoTokenizer, UMT5EncoderModel
        
        print("✅ 成功导入 WanProcessor")
        print(f"✅ Processor 属性: {WanProcessor.attributes}")
        
        # 测试清理函数
        from verl.utils.wan_processor import prompt_clean
        
        test_text = "  Hello   world!  \n\n  "
        cleaned = prompt_clean(test_text)
        print(f"✅ 文本清理测试: '{test_text}' -> '{cleaned}'")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_processor_attributes()
    test_wan_processor()  # 需要实际的模型路径才能运行 