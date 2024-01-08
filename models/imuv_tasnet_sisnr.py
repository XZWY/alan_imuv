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

    # remember to run: export PYTHONPATH=<working direcoty>/alan_imuv
    import os
    import sys
    from utils import load_checkpoint

    # set work_dir
    work_dir = '/home/exx/Documents/NAS/Datasets/alanweiyang'

    # inputs
    input = torch.rand(1, 32000) # input mixture, 16kHz sampling rate
    subband_clean_input = torch.rand(1, 32000) # subband input, 16kHz sampling rate, but only contains 800Hz effective(real) frequency

    # checkpoint dir
    ckpt_dir = os.path.join(work_dir, 'alan_imuv/models/ckpt_00130000')

    # normalize the samples so that it matches training
    training_max = 0.4003
    current_max = input.max()
    alpha = training_max / current_max
    input = alpha * input
    subband_clean_input = alpha * subband_clean_input

    # instantiate model and load weights
    model = IMUV_TASNET_SISNR()
    state_dict = load_checkpoint(ckpt_dir, input.device)
    model.load_state_dict(state_dict['model'])

    # running inference
    output = model.inference(input, subband_clean_input) # 1, n_samples
    print(output.shape)



if __name__ == "__main__":
    test()