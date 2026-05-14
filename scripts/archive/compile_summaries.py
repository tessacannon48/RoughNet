#!/usr/bin/env python3
"""Compile patch reconstruction summaries from all 18 model-sampler-region combinations."""

import json
import os

def main():
    base = os.path.join(os.path.dirname(__file__), '..', 'figures')
    models = ['cosine', 'linear']
    regions = ['pondinlet', 'tuk', 'cambridge']
    samplers = ['ddpm', 'ddim', 'plms']

    compiled = {
        'description': 'Compiled patch reconstruction summaries from all 18 model-sampler-region combinations',
        'models': models,
        'regions': regions,
        'samplers': samplers,
        'results': {}
    }

    for model in models:
        compiled['results'][model] = {}
        for region in regions:
            compiled['results'][model][region] = {}
            prefix = 'final_val' if region in ('pondinlet', 'tuk') else 'final_test'
            for sampler in samplers:
                folder = os.path.join(base, f'{model}_model', f'{prefix}_{region}_{sampler}', 'reconstruction_statistics')
                json_path = os.path.join(folder, 'patch_reconstruction_summary_demeaned.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    compiled['results'][model][region][sampler] = {
                        'source_path': json_path,
                        'num_patches': data.get('num_patches'),
                        'macro_average': data.get('macro_average'),
                        #'weighted_by_valid_pixel_count': data.get('weighted_by_valid_pixel_count')
                    }
                    print(f'Loaded: {json_path}')
                else:
                    compiled['results'][model][region][sampler] = {'error': f'File not found: {json_path}'}
                    print(f'MISSING: {json_path}')

    out_dir = os.path.join(base, 'ablation_comparison')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'compiled_patch_summaries_demeaned.json')
    with open(out_path, 'w') as f:
        json.dump(compiled, f, indent=2)
    print(f'\nCompiled file saved to: {out_path}')

if __name__ == '__main__':
    main()
