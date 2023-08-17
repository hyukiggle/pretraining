import argparse
import yaml

def load_yaml(file_path):
    with open(file_path, 'r') as config_file:
        return yaml.safe_load(config_file)


if __name__== '__main__':
    filepath = '/home/hyukiggle/Documents/workspace/pretrain/reconstruction/utils/swin_tiny_patch4_window7_224_lite.yaml'
    
    parser = argparse.ArgumentParser(description="ImageNet pretraining")
    parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--epochs", default=150, type = int)
    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--num_steps", default=500000, type=int, help='number of training iterations')
    parser.add_argument("--warmup_steps", default = 500, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--opts", default='adamw', type=str)
    parser.add_argument("--lrdecay", action="store_true", help="enable learing rate scheduling")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--noamp", action="store_true")
    parser.add_argument("--n_gpu", default=1, type=int)
    parser.add_argument("--root_dir", default = "/home/hyukiggle/Documents/data/ImageNet", type=str)
    
    parser.add_argument("--config", default='/home/hyukiggle/Documents/workspace/pretrain/reconstruction/utils/swin_tiny_patch4_window7_224_lite.yaml',
                        type = str, help = "Path to config file")
    
    args = parser.parse_args()
    print(args)
    config = load_yaml(args.config)
    
    print(config)