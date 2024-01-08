import torch
import torch.nn as nn

from models.networks.convtasnet import TasNet
from models.networks.conditional_convtasnet import conditionalTasNet
from models.loss.sisnr_loss import calc_sdr_torch



# conditional_tasnet with sisnr loss
class IMUV_TASNET_SISNR(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=128, sr=16000, win=2, layer=8, stack=3, 
                 kernel=3, num_spk=2, causal=False):
        super(IMUV_TASNET_SISNR, self).__init__()
        
        self.network = conditionalTasNet()
    
    def inference(self, input, conditional_input):
        return self.network(input, conditional_input).squeeze(1)

    def loss(self, output, reference):
        return -calc_sdr_torch(output, reference)

    def forward(self, batch):
        output = self.inference(batch['mixture'], batch['target_sb_noisy'])
        loss = self.loss(output, batch['target'])

        batch['output'] = output
        batch['loss'] = loss

        return batch

def test():

    from utils import load_checkpoint

    ckpt_dir = '/home/exx/Documents/NAS/Datasets/alanweiyang/alan_imuv/models/ckpt_00130000'
    input = torch.rand(2, 32000) # input mixture
    subband_clean_input = torch.rand(2, 32000) # subband input

    model = IMUV_TASNET_SISNR()
    state_dict = load_checkpoint(ckpt_dir, input.device)
    model.load_state_dict(state_dict['model'])
    output = model.inference(input, subband_clean_input)
    print(output.shape)



if __name__ == "__main__":
    test()