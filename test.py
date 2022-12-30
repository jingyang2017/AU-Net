import argparse
import torch
from data.data_load import data_load
from models.aunet import AU_NET
from torchvision import transforms
from utils.evaluation import evaluate_multi
from utils.metrics import get_acc, get_f1
import logging
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='model evaluation')
parser.add_argument('--path', type=str)
parser.add_argument('--data', type=str)
parser.add_argument('--subset', type=int)

args = parser.parse_args()

if args.data == 'bp4d':
    model = AU_NET(alpha=0.9, beta=0.1, n_classes=12)
elif args.data == 'disfa':
    model = AU_NET(alpha=0.9, beta=0.1, n_classes=8)
else:
    raise NotImplementedError()

pre_trained = torch.load(args.path, 'cpu')
pretrained_items = list(pre_trained.items())
current_items = model.predictor.state_dict()
count = 0
for key, value in current_items.items():
    layer_name, weights = pretrained_items[count]
    current_items[key] = weights
    count = count + 1
model.predictor.load_state_dict(current_items, strict=True)
model = model.cuda()
model.eval()

transform_valid = transforms.Compose([transforms.ToTensor()])
val_data = data_load(args.data, phase='test', subset=args.subset,  flip=False,transform=transform_valid,seed=0)
test_loader_1 = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

val_data = data_load(args.data, phase='test', subset=args.subset,  flip=True,transform=transform_valid,seed=0)
test_loader_2 = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
output = evaluate_multi(model, test_loader_1, test_loader_2, metrics={'ACC': get_acc, 'F1': get_f1})

print('f1 score: ', str(output['F1'].numpy().tolist()))
print('average f1 score: ', str(output['F1'].mean().numpy().tolist()))



