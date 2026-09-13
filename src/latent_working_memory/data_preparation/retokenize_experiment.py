"""使用原始字符跨度重新分词，保留已有实验样本与 split。"""

import argparse
import json
import uuid
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from latent_working_memory.data_preparation.fineweb import data_contract
from latent_working_memory.v1.config import load_config
from latent_working_memory.v1.data import Episode
from latent_working_memory.v1.sampling import capacity_weights, read_tokens


def prepare(spec, config, tokenizer, output):
    output.mkdir(parents=True, exist_ok=False)
    documents = {}
    with Path(spec['source_pool']).open() as handle:
        for line in handle:
            entry = json.loads(line)
            if entry['status'] == 'eligible':
                documents[entry['record']['id']] = (entry['record']['text'], entry['split'])
    report = {'source': spec, 'datasets': {}}
    for name, splits in spec['datasets'].items():
        directory = output / name
        directory.mkdir()
        counts, source_preparations, statistics = {}, {}, {}
        for split, source_path in splits.items():
            source_path = Path(source_path)
            parent_metadata = json.loads((source_path.parent / 'preparation.json').read_text())
            source_preparations[split] = parent_metadata['preparation_id']
            kept, rejected, cells = 0, Counter(), Counter()
            with source_path.open() as inp, (directory / f'{split}.jsonl').open('w') as out:
                for line in inp:
                    row = json.loads(line)
                    source = row['sources'][0]
                    provenance = source['provenance']
                    text, document_split = documents[source['document_id']]
                    if document_split != split:
                        raise ValueError('document split differs from source pool')
                    start, end = provenance['x_char_span']
                    x = text[start:end]
                    read = row['reads'][0]
                    target_span = provenance['y_char_span']
                    target = text[slice(*target_span)] if target_span else x
                    if target != read['references'][0]['text']:
                        raise ValueError('reference differs from original character span')
                    ids = tokenizer.encode(x, add_special_tokens=False)
                    row['input_ids'] = ids
                    row['write_ends'] = [len(ids)]
                    source['token_end'] = len(ids)
                    read['prefix_end'] = len(ids)
                    provenance['tokenizer_name_or_path'] = config.model_name_or_path
                    provenance['tokenizer_revision'] = config.model_revision
                    episode = Episode.from_record(row)
                    ae, lm = read_tokens(episode, tokenizer)
                    capacities = capacity_weights(config, len(ids), ae, lm, 0)
                    if not capacities:
                        rejected['length_or_capacity'] += 1
                        continue
                    task = ae if ae is not None else lm
                    if 1 + len(ids) + len(task.prompt_ids) + len(task.target_ids) > config.read_context_tokens:
                        rejected['full_context_window'] += 1
                        continue
                    bucket = next(b for b in config.input_length_bounds if len(ids) <= b)
                    cells[f"{provenance['boundary_variant']}/{read['task']}/{bucket}"] += 1
                    out.write(json.dumps(row, ensure_ascii=False) + '\n')
                    kept += 1
            counts[split] = kept
            statistics[split] = {'retained': kept, 'rejected': dict(rejected), 'task_source_length': dict(cells)}
            print(name, split, kept, dict(rejected), flush=True)
        metadata = {'preparation_id': str(uuid.uuid4()), 'contract': data_contract(config),
                    'source_preparations': source_preparations, 'counts': counts,
                    'source_weights': parent_metadata['source_weights']}
        (directory / 'preparation.json').write_text(json.dumps(metadata, indent=2) + '\n')
        report['datasets'][name] = statistics
    report['evaluation_dirs'] = {n: str((output / n).resolve()) for n in spec['datasets'] if n != 'mixed'}
    (output / 'retokenization.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, local_files_only=True)
    prepare(json.loads(args.spec.read_text()), config, tokenizer, args.output_dir)


if __name__ == '__main__':
    main()
