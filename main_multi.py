import os
import argparse
import torch
import yaml
from torchvision import datasets
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import Multi_ResNet18
from trainer_multi import BYOLTrainer
import torchvision
import torchvision.transforms as transforms

print(torch.__version__)
torch.manual_seed(0)


def main(args):
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    data_transform = get_simclr_data_transforms(**config['data_transforms'])
    train_dataset_1 = datasets.CIFAR10(root='../STL_model/data', train=True, download=True,
                                       transform=MultiViewDataInjector([data_transform, data_transform]))

    data_transforms = torchvision.transforms.Compose([transforms.Resize(96),
                                                      transforms.ToTensor()])
    train_dataset_2 = datasets.CIFAR10(root='../STL_model/data', train=True, download=True,
                                       transform=MultiViewDataInjector([data_transforms, data_transforms]))

    # online network
    online_network = Multi_ResNet18(args.flag_ova, **config['network']).to(device)
    pretrained_folder = config['network']['fine_tune_from']

    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(device)))

            online_network.load_state_dict(load_params['online_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = Multi_ResNet18(args.flag_ova, **config['network']).to(device)

    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          model_path=args.model_path,
                          **config['trainer'])

    trainer.train((train_dataset_1, train_dataset_2), args.flag_ova)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path of the model', default='save/temp.pth')
    parser.add_argument('--flag_ova', action='store_true', help='train ova or not', default=False)
    args = parser.parse_args()
    
    main(args)
