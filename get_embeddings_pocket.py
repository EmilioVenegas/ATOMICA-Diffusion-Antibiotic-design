import argparse
from ATOMICA.get_embeddings_masking import main as get_embeddings_main 



args = argparse.Namespace(
    model_config='ATOMICA/pretrain/pretrain_model_config.json',
    model_weights='ATOMICA/pretrain/pretrain_model_weights.pt',
    data_path='ATOMICA/data/example/example_outputs.pkl',
    output_path='ATOMICA/data/example/example_outputs_embedded.pkl', 
    batch_size=2,
    model_ckpt=None
)


get_embeddings_main(args)
