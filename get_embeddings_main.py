import argparse
from ATOMICA.get_embeddings import main as get_embeddings_main 



args = argparse.Namespace(
    model_config='pretrain/pretrain_model_config.json',
    model_weights='pretrain/pretrain_model_weights.pt',
    data_path='data/example/example_outputs.pkl',
    output_path='data/example/example_outputs_embedded.pkl', 
    batch_size=4,
    model_ckpt=None
)


get_embeddings_main(args)
