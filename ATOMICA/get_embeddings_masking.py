from tqdm import tqdm
import pickle
from ATOMICA.data.dataset import PDBDataset, ProtInterfaceDataset
from ATOMICA.models.prediction_model import PredictionModel
from ATOMICA.models.pretrain_model import DenoisePretrainModel
from ATOMICA.models.prot_interface_model import ProteinInterfaceModel
from ATOMICA.trainers.abs_trainer import Trainer
import torch
import json
import numpy as np

def main(args):
    if args.model_ckpt:
        model = torch.load(args.model_ckpt)
    elif args.model_config and args.model_weights:
        with open(args.model_config, "r") as f:
            model_config = json.load(f)
        if model_config['model_type'] == 'PredictionModel' or model_config['model_type'] == 'DenoisePretrainModel':
            model = PredictionModel.load_from_config_and_weights(args.model_config, args.model_weights)
        elif model_config['model_type'] == 'ProteinInterfaceModel':
            model = ProteinInterfaceModel.load_from_config_and_weights(args.model_config, args.model_weights)
        else:
            raise NotImplementedError(f"Model type {model_config['model_type']} not implemented")

    if isinstance(model, ProteinInterfaceModel):
        print("Model is ProteinInterfaceModel, extracting prot_model.")
        model = model.prot_model
        dataset = ProtInterfaceDataset(args.data_path)
    else:
        dataset = PDBDataset(args.data_path)
    
    if isinstance(model, DenoisePretrainModel) and not isinstance(model, PredictionModel):
        model = PredictionModel.load_from_pretrained(args.model_ckpt)
    model = model.to("cuda")
    batch_size = args.batch_size
    POCKET_MASK_KEY = 'segment_ids'

    embeddings = []
    for idx in tqdm(range(0, len(dataset), batch_size), desc="Embedding data", total=len(dataset)//batch_size+1):
        items = dataset.data[idx:min(idx+batch_size, len(dataset))]

        outputs = []
        try:
            for item in items:
                outputs.append({"id": item["id"]})
            if isinstance(dataset, ProtInterfaceDataset):
                batch_items = [item["prot_data"] for item in items]
            else:
                batch_items = [item["data"] for item in items]
            batch = PDBDataset.collate_fn(batch_items)
            batch = Trainer.to_device(batch, "cuda")
            return_obj = model.infer(batch)
            
            curr_block = 0
            curr_atom = 0
            for i, item in enumerate(items):
                # Get the correct data source
                data_source = item["data"] if not isinstance(dataset, ProtInterfaceDataset) else item["prot_data"]
                
                num_blocks = len(data_source["B"])
                num_atoms = len(data_source["A"])

                # 1. Get ALL embeddings and IDs for this item
                all_block_embeddings = return_obj.block_repr[curr_block: curr_block + num_blocks].detach().cpu().numpy()
                all_atom_embeddings = return_obj.unit_repr[curr_atom: curr_atom + num_atoms].detach().cpu().numpy()
                all_block_ids = np.array(data_source["B"])
                all_atom_ids = np.array(data_source["A"])

                # 2. Save the graph-level embedding
                outputs[i]["graph_embedding"] = return_obj.graph_repr[i].detach().cpu().numpy()

                if POCKET_MASK_KEY in data_source:
                    # 3. Create the pocket mask
                    # Get the segment IDs (a list of 0s and 1s)
                    segment_ids_mask = np.array(data_source[POCKET_MASK_KEY])
                    
                    # Create a boolean mask: True for protein (0), False for ligand (1)
                    pocket_block_mask = (segment_ids_mask == 0)
                    
                    # 4. Filter block embeddings and IDs using the mask
                    outputs[i]["block_embedding"] = all_block_embeddings[pocket_block_mask]
                    outputs[i]["block_id"] = all_block_ids[pocket_block_mask].tolist()

                    # 5. (Optional) Filter atom embeddings
                    atom_to_block_map_key = 'A_B' 
                    if atom_to_block_map_key in data_source:
                        atom_to_block_index = np.array(data_source[atom_to_block_map_key])
                        # Use the block mask to create an atom mask
                        pocket_atom_mask = pocket_block_mask[atom_to_block_index]
                        
                        outputs[i]["atom_embedding"] = all_atom_embeddings[pocket_atom_mask]
                        outputs[i]["atom_id"] = all_atom_ids[pocket_atom_mask].tolist()
                    else:
                        print(f"Warning: '{atom_to_block_map_key}' not found for {item['id']}. Cannot filter atom embeddings.")
                        outputs[i]["atom_embedding"] = np.array([])
                        outputs[i]["atom_id"] = []
                
                else:
                    # Fallback if key is missing
                    print(f"Warning: Mask key '{POCKET_MASK_KEY}' not found in item {item['id']}. Saving all embeddings.")
                    outputs[i]["block_embedding"] = all_block_embeddings
                    outputs[i]["atom_embedding"] = all_atom_embeddings
                    outputs[i]["block_id"] = all_block_ids.tolist()
                    outputs[i]["atom_id"] = all_atom_ids.tolist()
                
                # --- END OF MODIFICATION ---

                curr_block += num_blocks
                curr_atom += num_atoms
        except Exception as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                print("CUDA out of memory, reducing batch size to 1 for this batch.")
                outputs = []
                # go through the batch one by one
                for item in items:
                    try:
                        # Define data_source for the CURRENT item in the loop
                        data_source = item["data"] if not isinstance(dataset, ProtInterfaceDataset) else item["prot_data"]
                        output = {"id": item["id"]}
                        batch = PDBDataset.collate_fn([item["data"] if not isinstance(dataset, ProtInterfaceDataset) else item["prot_data"]])
                        batch = Trainer.to_device(batch, "cuda")
                        return_obj = model.infer(batch)
                        # --- MODIFIED OOM BLOCK ---
                        all_block_embeddings = return_obj.block_repr.detach().cpu().numpy()
                        all_atom_embeddings = return_obj.unit_repr.detach().cpu().numpy()
                        all_block_ids = np.array(data_source["B"])
                        all_atom_ids = np.array(data_source["A"])

                        output["graph_embedding"] = return_obj.graph_repr[0].detach().cpu().numpy()

                        if POCKET_MASK_KEY in data_source:
                            # Create the mask
                            segment_ids_mask = np.array(data_source[POCKET_MASK_KEY])
                            pocket_block_mask = (segment_ids_mask == 0)
                            
                            # Filter blocks
                            output["block_embedding"] = all_block_embeddings[pocket_block_mask]
                            output["block_id"] = all_block_ids[pocket_block_mask].tolist()

                            # Filter atoms
                            atom_to_block_map_key = 'A_B' 
                            if atom_to_block_map_key in data_source:
                                atom_to_block_index = np.array(data_source[atom_to_block_map_key])
                                pocket_atom_mask = pocket_block_mask[atom_to_block_index]
                                
                                output["atom_embedding"] = all_atom_embeddings[pocket_atom_mask]
                                output["atom_id"] = all_atom_ids[pocket_atom_mask].tolist()
                            else:
                                output["atom_embedding"] = np.array([])
                                output["atom_id"] = []
                        
                        else:
                            # Fallback
                            output["block_embedding"] = all_block_embeddings
                            output["atom_embedding"] = all_atom_embeddings
                            output["block_id"] = all_block_ids.tolist()
                            output["atom_id"] = all_atom_ids.tolist()
                        # --- END OF MODIFIED OOM BLOCK ---
                    
                        outputs.append(output)
                    except Exception as e:
                        print(f"Error processing item {item['id']}: {e}")
                        torch.cuda.empty_cache()
                        continue
            else:
                import pdb; pdb.set_trace()
                raise e
        embeddings.extend(outputs)
    
    with open(args.output_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saving processed data to {args.output_path}. Total of {len(embeddings)} items.")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt', type=str, default=None, help='path of the model ckpt to load')
    parser.add_argument('--model_config', type=str, default=None, help='path of the model config to load')
    parser.add_argument('--model_weights', type=str, default=None, help='path of the model weights to load')
    parser.add_argument("--output_path", type=str, required=True, help='Path to save the output embeddings, should be a .pkl file')
    parser.add_argument("--data_path", type=str, required=True, help='Path to the data file either in json or pickle format')
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)