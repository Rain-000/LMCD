import argparse
import test_Data
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import CombinedModel
import torch.nn.functional as F
import logging


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='Model evaluation script')
parser.add_argument('--file_path', type=str, required=True, help='Path to the input text file')
args = parser.parse_args()


logging.basicConfig(filename='result.log', filemode='w', level=logging.INFO, format='%(message)s')
logger = logging.getLogger()


model = CombinedModel().to(DEVICE)
model.load_state_dict(torch.load('Model1.pth', map_location=DEVICE))
model.eval()


file_path = args.file_path
data_processor = test_Data.DataProcessor(file_path)

x, ids = data_processor.process_data()
X_test = x.astype(np.float32)


test_dataset = TensorDataset(torch.tensor(X_test))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


with torch.no_grad():
    for idx, (x,) in enumerate(test_loader):
        x = x.to(DEVICE)
        x = x.permute(0, 2, 1)
        y_pred = model(x)
        probabilities = F.softmax(y_pred, dim=1)
        _, predicted = torch.max(y_pred, 1)

        for i in range(x.size(0)):
            sample_id = ids[idx * test_loader.batch_size + i]
            pred_label = predicted[i].item()
            prob = probabilities[i, pred_label].item()


            result = f'ID: {str(sample_id)}, Label: {str(pred_label)}, Prob: {prob:.4f}'
            print(result)
            logger.info(result)
