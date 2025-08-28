#!/usr/bin/env python3
"""Script to explore the GAIA dataset"""

from datasets import load_dataset
import json

def explore_gaia_dataset():
    """Explore the GAIA dataset and show examples"""
    
    print("Loading GAIA dataset...")
    
    # Load different levels
    datasets = {}
    for level in ['2023_level1', '2023_level2', '2023_level3', '2023_all']:
        try:
            datasets[level] = load_dataset('gaia-benchmark/GAIA', name=level)
            print(f"✓ Loaded {level}")
        except Exception as e:
            print(f"✗ Failed to load {level}: {e}")
    
    # Explore the first dataset
    if '2023_level1' in datasets:
        dataset = datasets['2023_level1']
        
        print(f"\nDataset structure:")
        print(f"Available splits: {list(dataset.keys())}")
        
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            print(f"\n{split_name.upper()} split:")
            print(f"  Number of examples: {len(split_data)}")
            
            # Show first example
            if len(split_data) > 0:
                first_example = split_data[0]
                print(f"  First example:")
                print(f"    Task ID: {first_example['task_id']}")
                print(f"    Question: {first_example['Question'][:100]}...")
                print(f"    Level: {first_example['Level']}")
                print(f"    File name: {first_example['file_name']}")
                print(f"    File path: {first_example['file_path']}")
                
                # Show metadata if available
                if 'Annotator Metadata' in first_example:
                    metadata = first_example['Annotator Metadata']
                    print(f"    Metadata: {metadata}")
    
    # Show statistics across all levels
    print(f"\n{'='*50}")
    print("DATASET STATISTICS")
    print(f"{'='*50}")
    
    for level_name, dataset in datasets.items():
        print(f"\n{level_name}:")
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            print(f"  {split_name}: {len(split_data)} examples")
            
            # Count levels in this dataset
            if len(split_data) > 0:
                levels = {}
                for example in split_data:
                    level = example['Level']
                    levels[level] = levels.get(level, 0) + 1
                print(f"    Level distribution: {levels}")

if __name__ == "__main__":
    explore_gaia_dataset() 