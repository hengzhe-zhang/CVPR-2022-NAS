from os import listdir

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from dataset_loader import arch_list_train, train_list
from example.custom_vit import ViT
from example.data_utils import read_image, image_preprocess
from utils.learning_utils import kendalltau
from utils.notify_utils import notify

cuda = torch.device('cuda')  # Default CUDA device

heads = arch_list_train[:, 1::2]
ratios = arch_list_train[:, 2::2]
heads[heads == 1] = 12
heads[heads == 2] = 11
heads[heads == 3] = 10
ratios[heads == 1] = 4
ratios[heads == 3] = 3
ratios[heads == 2] = 3.5

all_data = []
for data in listdir('example/data/cplfw'):
    image = read_image(f'example/data/cplfw/{data}')
    image = np.array(image)
    image, _ = image_preprocess(image, 224)
    imga = to_tensor(image)
    imga = torch.unsqueeze(imga, 0)
    all_data.append(imga)
all_data = torch.concat(all_data[:16], axis=0)


def counting_forward_hook(module, inp, out):
    try:
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.view(inp.size(0), -1)
        x = (inp > 0).float()
        K = x @ x.t()
        K2 = (1. - x) @ (1. - x.t())
        network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
    except:
        raise Exception


def counting_backward_hook(module, inp, out):
    module.visited_backwards = True


@notify
def task():
    global network
    all_scores = []
    for id, depth, head, ratio in zip(range(len(arch_list_train)), arch_list_train[:50, 0], heads[:], ratios[:]):
        with torch.no_grad():
            network = ViT(
                image_size=all_data.shape[2] * all_data.shape[3],
                patch_size=16,
                num_classes=10,
                dim=768,
                depth=depth,
                heads=head,
                ratios=ratio
            )

            network.K = np.zeros((len(all_data), len(all_data)))

            for name, module in network.named_modules():
                if 'GELU' in str(type(module)):
                    # hooks[name] = module.register_forward_hook(counting_hook)
                    module.register_forward_hook(counting_forward_hook)
                    module.register_backward_hook(counting_backward_hook)

            preds = network(all_data)  # (1, 1000)
            score = np.linalg.slogdet(network.K)[1]
            all_scores.append(score)
            print(-1 * score, train_list[1][id])
    print(kendalltau(all_scores, np.array(train_list[1][:50])))


if __name__ == '__main__':
    task()
