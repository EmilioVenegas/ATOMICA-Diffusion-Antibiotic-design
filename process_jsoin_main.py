import argparse
from process_atomica_jsonl import main as process_json 



args = argparse.Namespace(
    input_file='PL.jsonl.gz',
    model_config='ATOMICA/pretrain/pretrain_model_config.json',
    model_weights='ATOMICA/pretrain/pretrain_model_weights.pt',
    output_dir='outputs_embedded.pt'
)


process_json(args)

