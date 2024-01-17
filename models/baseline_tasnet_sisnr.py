import torch
import torch.nn as nn

from models.networks.convtasnet import TasNet
from models.networks.conditional_convtasnet import conditionalTasNet
from models.loss.sisnr_loss import calc_sdr_torch, batch_SDR_torch



# conditional_tasnet with sisnr loss
class Baseline_TASNET_SISNR(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=128, sr=16000, win=2, layer=8, stack=3, 
                 kernel=3, num_spk=2):
        super(Baseline_TASNET_SISNR, self).__init__()
        
        self.network = TasNet(enc_dim=enc_dim, feature_dim=feature_dim, sr=sr, win=win, layer=layer, stack=stack, 
                 kernel=kernel, num_spk=num_spk)
    
    def inference(self, input):
        return self.network(input) # bs, num_sources, n_samples

    def loss(self, output, reference):
        return -batch_SDR_torch(output, reference)

    def forward(self, batch):
        output = self.inference(batch['mixture'])
        loss = self.loss(output, batch['sources'])

        batch['output'] = output
        batch['loss'] = loss

        return batch


def test():

    # remember to run: export PYTHONPATH=<working direcoty>/alan_imuv
    import os
    import sys
    from utils import load_checkpoint
    import soundfile as sf
    from models.loss.sisnr_loss import calc_sdr_torch, batch_SDR_torch

    # set work_dir
    # work_dir = '/workspace/host'
    work_dir = '/home/exx/Documents/NAS/Datasets/alanweiyang'

    # inputs from dataset or test random tensors, if dataset is wanted, then need to get into docker alan_imuv by docker exec -it alan_imuv bash, otherwise comment below two lines for testing
    # from dataset import LibriMixIMUV, LibriMix, collate_func_separation
    # dataset = LibriMix(csv_dir='/workspace/host/LibriMix/dataset/Libri2Mix/wav16k/min/metadata', sample_rate=16000, n_src=2, segment=5, return_id=False, mode='dev')
    # batch = dataset[2]
    # input = batch['mixture'].unsqueeze(0)
    # mixture = batch['mixture']
    # # assume target is source 1
    # sources = batch['sources']
    # target = sources[1]

    # if the dataset is not available then just testing on random data
    input = torch.randn(1, 16000)
    sources = torch.randn(2, 16000)
    mixture = torch.randn(16000)
    target = torch.randn(16000)

    # checkpoint dir
    ckpt_dir = os.path.join(work_dir, 'alan_imuv/models/ckpt_00570000') # load small model

    # instantiate model and load weights
    # model = Baseline_TASNET_SISNR(enc_dim=512, feature_dim=128, sr=16000, win=2, layer=8, stack=3, kernel=3, num_spk=2) # large model ~5M parameters
    model = Baseline_TASNET_SISNR(enc_dim=128, feature_dim=64, sr=16000, win=2, layer=4, stack=1, kernel=3, num_spk=2) # small model ~270K parameters

    state_dict = load_checkpoint(ckpt_dir, input.device)
    model.load_state_dict(state_dict['model'])

    torch.set_printoptions(precision=10)

    # set device
    model = model.cuda()
    mixture = mixture.cuda()
    input = input.cuda()
    target = target.cuda()
    sources = sources.cuda()

    # running inference
    with torch.no_grad():
        output = model.inference(input) # 1, 2, n_samples
    output = output.squeeze(0)

    # rescale for sisnr loss
    alpha_mix = (output * mixture).sum(1) / (output**2).sum().clamp_min(1e-8)
    output = alpha_mix.unsqueeze(1) * output # output 2 sources (2, n_samples)

    # need to find the max snr source
    snrs = 10 * torch.log10(target.square().sum() / (output-target).square().sum(1))
    idx = torch.argmax(snrs)
    final_output = output[idx] # n_samples    
    print('final snr', 10 * torch.log10(target.square().sum() / (final_output-target).square().sum()))


if __name__ == "__main__":
    test()