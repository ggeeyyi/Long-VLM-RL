#!/usr/bin/env python3
"""
Script to convert ScienceQA dataset to HuggingFace format with columns: images, problem, answer
Split into train and test subsets (1.2k test, rest train) and push to HuggingFace Hub as GY2233/lmms-ScienceQA
"""

import os
import random
import io
from PIL import Image as PILImage
from datasets import load_dataset, Dataset, Features, Image, Value, DatasetDict, Sequence
from huggingface_hub import login
from tqdm import tqdm

def convert_scienceqa_to_hf():
    """Convert ScienceQA dataset to HuggingFace format and push to hub"""
    
    # Create a dummy image for cases where image is None
    dummy_image = PILImage.new('RGB', (224, 224), color='white')
    
    # Load the original ScienceQA dataset
    print("Loading ScienceQA-FULL dataset...")
    try:
        ds = load_dataset("lmms-lab/ScienceQA", "ScienceQA-FULL")
        print(f"Loaded dataset with splits: {list(ds.keys())}")
        print(f"Dataset info: {ds}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("You may need to authenticate with HuggingFace first")
        return
    
    # Define the features for the new dataset
    features = Features({
        "images": Sequence(Image()),
        "problem": Value("string"),
        "answer": Value("string"),
        "mask_image": Value("bool"),
    })
    
    # Combine all splits into one dataset
    print("\nCombining all splits...")
    all_examples = []
    
    for split_name, split_data in ds.items():
        print(f"Processing {split_name} split with {len(split_data)} examples...")
        
        # Process each example in this split
        for example in tqdm(split_data, desc=f"Converting {split_name}"):
            # Extract image
            image = example.get("image")
            
            # Use dummy image if image is None, otherwise convert to RGB if needed
            is_dummy_image = False
            if image is None:
                image = dummy_image
                is_dummy_image = True
            elif hasattr(image, 'convert'):
                image = image.convert("RGB")
            
            # Create list of images (ScienceQA has single image per example)
            images = [image]
            # Format the problem text
            context = example.get("hint", "")
            question = example.get("question", "")
            choices = example.get("choices", [])
            
            # Create choice options (A, B, C, D)
            len_choices = len(choices)
            options = [chr(ord("A") + i) for i in range(len_choices)]
            choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
            
            # Format the full problem
            if context:
                problem = f"<image>Context: {context}\n\nQuestion: {question}\n\nOptions:\n{choices_str}"
            else:
                problem = f"<image>Question: {question}\n\nOptions:\n{choices_str}"
            
            # Convert numeric answer to letter
            answer_idx = example.get("answer", 0)
            answer = chr(ord('A') + answer_idx)
            
            # Add to combined list
            all_examples.append({
                "images": images,
                "problem": problem,
                "answer": answer,
                "mask_image": is_dummy_image
            })
    
    print(f"\nTotal examples collected: {len(all_examples)}")
    
    # Show statistics about dummy images
    dummy_count = sum(1 for ex in all_examples if ex["mask_image"])
    print(f"Examples with dummy images: {dummy_count} ({dummy_count/len(all_examples)*100:.2f}%)")
    print(f"Examples with real images: {len(all_examples) - dummy_count} ({(len(all_examples) - dummy_count)/len(all_examples)*100:.2f}%)")
    
    # Shuffle the data for random split
    random.seed(42)  # For reproducibility
    random.shuffle(all_examples)
    
    # Split into test (1200) and train (rest)
    test_size = 1500
    test_examples = all_examples[:test_size]
    train_examples = all_examples[test_size:]
    
    print(f"Split into:")
    print(f"  Test: {len(test_examples)} examples")
    print(f"  Train: {len(train_examples)} examples")
    
    # Create datasets for each split
    def create_dataset_from_examples(examples):
        return Dataset.from_dict({
            "images": [ex["images"] for ex in examples],
            "problem": [ex["problem"] for ex in examples],
            "answer": [ex["answer"] for ex in examples],
            "mask_image": [ex["mask_image"] for ex in examples]
        }, features=features)
    
    converted_splits = {
        "train": create_dataset_from_examples(train_examples),
        "test": create_dataset_from_examples(test_examples)
    }
    
    # Create the final dataset
    if converted_splits:
        final_dataset = DatasetDict(converted_splits)
        
        print(f"\nFinal dataset structure:")
        print(final_dataset)
        
        # Show examples from both splits
        print(f"\nExample from train split:")
        example = final_dataset["train"][0]
        print(f"Problem: {example['problem'][:200]}...")
        print(f"Answer: {example['answer']}")
        print(f"Mask Image: {example['mask_image']}")
        
        print(f"\nExample from test split:")
        example = final_dataset["test"][0]
        print(f"Problem: {example['problem'][:200]}...")
        print(f"Answer: {example['answer']}")
        print(f"Mask Image: {example['mask_image']}")
        
        # Push to HuggingFace Hub
        print(f"\nPushing to HuggingFace Hub as GY2233/lmms-ScienceQA...")
        try:
            final_dataset.push_to_hub("GY2233/lmms-ScienceQA", token=True)
            print("Successfully pushed to HuggingFace Hub!")
        except Exception as e:
            print(f"Error pushing to hub: {e}")
            print("You may need to authenticate with HuggingFace first")
            print("Run: huggingface-cli login")
    else:
        print("No data converted!")

if __name__ == "__main__":
    convert_scienceqa_to_hf() 