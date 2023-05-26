import torch

#To allow plotting pytorch tensors
torch.Tensor.ndim = property(lambda self: len(self.shape))

N_CLASSES = 2

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

from packages.CNNLSTM import CNNLSTM

CRNN_Windowed_DeltaMFCC = CNNLSTM(
    input_size=42, n_classes=2, n_layers_rnn=64, fc_in=9408, device='cpu', in_channels=3, delta_spec=False
)

myModel = CRNN_Windowed_DeltaMFCC

checkpoint = torch.load('model_checkpoint.pth', map_location=torch.device('cpu'))
myModel.load_state_dict(checkpoint['model_state_dict'])

# signal level evaluation
def ClassifySignal(cycles_results, tr=1.25):
    """Combines the binary classification results for each cycle using a voting system."""
    num_ones = sum(cycles_results)
    if num_ones > tr * len(cycles_results)/2:   
        return 1
    else:
        return 0

def getPrediction(signals):
    preds = []
    myModel.eval()
    with torch.inference_mode():
        for sig in signals:
            output = myModel(sig.unsqueeze(dim=0))
            _, prediction = torch.max(output, 1)
            preds.append(prediction.item())
    num_ones = sum(preds)
    result = num_ones/len(preds)
    return result