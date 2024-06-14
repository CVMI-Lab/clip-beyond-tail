import os
import argparse
import torch

def load_pretrained_weights(pretrained_weights, checkpoint_key="teacher"):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # remove items with leading "head."
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.") and not k.startswith("clip_model.")}
        state_dict = {"state_dict": state_dict}
        return state_dict
    return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DINO checkpoint to timm format")
    parser.add_argument("--model", type=str, help="path to the pretrained weights")
    parser.add_argument("--output_path", type=str, default=None, help="path to the output checkpoint")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = os.path.join(os.path.dirname(args.model), "dino.pth")
    
    state_dict = load_pretrained_weights(args.model)
    if state_dict != {}:
        torch.save(state_dict, args.output_path)
        print(f"Converted checkpoint saved to {args.output_path}")