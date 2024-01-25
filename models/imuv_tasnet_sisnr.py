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

# def test():

#     # remember to run: export PYTHONPATH=<working direcoty>/alan_imuv
#     import os
#     import sys
#     from utils import load_checkpoint
#     import soundfile as sf

#     # set work_dir
#     work_dir = '/workspace/host'

#     # inputs
#     from dataset import LibriMixIMUV
#     dataset = LibriMixIMUV(
#         csv_dir='/workspace/host/LibriMix/dataset/Libri3Mix/wav16k/min/metadata',
#         sample_rate=16000,
#         noisy=True,
#         min_num_sources=1,
#         max_num_sources=3,
#         subband_snr_range=[3, 10],
#         # segment=5,
#         mode='dev'     
#     )
#     batch = dataset[2]
#     input = batch['mixture'].unsqueeze(0)
#     subband_clean_input = batch['target_sb_noisy'].unsqueeze(0)
#     # input = torch.rand(1, 32000) # input mixture, 16kHz sampling rate
#     # subband_clean_input = torch.rand(1, 32000) # subband input, 16kHz sampling rate, but only contains 800Hz effective(real) frequency

#     # checkpoint dir
#     ckpt_dir = os.path.join(work_dir, 'alan_imuv/models/ckpt_imu_tasnet')

#     # normalize the samples so that it matches training
#     training_max = 0.4003
#     current_max = input.max()
#     alpha = training_max / current_max
#     input = alpha * input
#     subband_clean_input = alpha * subband_clean_input

#     # instantiate model and load weights
#     model = IMUV_TASNET_SISNR()
#     state_dict = load_checkpoint(ckpt_dir, input.device)
#     model.load_state_dict(state_dict['model'])

#     torch.set_printoptions(precision=10)
#     # running inference
#     with torch.no_grad():
#         output = model.inference(input, subband_clean_input) # 1, n_samples
#     print(output.shape)

#     target = batch['target']
#     mixture = batch['mixture']
#     output = output.squeeze(0)
#     alpha = (output * target).sum() / (output**2).sum()
#     alpha_mix = (output * mixture).sum() / (output**2).sum()
#     print(alpha, alpha_mix)

#     print(10 * torch.log10(target.square().sum() / (alpha*output-target).square().sum()))
#     print(10 * torch.log10(target.square().sum() / (alpha_mix*output-target).square().sum()))
#     print(10 * torch.log10(target.square().sum() / (mixture-target).square().sum()))

#     sf.write('mixture.wav', batch['mixture'], 16000)
#     sf.write('target.wav', batch['target'], 16000)
#     sf.write('target_sb.wav', batch['target_sb'], 16000)
#     sf.write('target_sb_noisy.wav', batch['target_sb_noisy'], 16000)
#     sf.write('output.wav', 0.0004*output, 16000)

def test():

    # remember to run: export PYTHONPATH=<working direcoty>/alan_imuv
    import os
    import sys
    from utils import load_checkpoint
    import soundfile as sf

    # set work_dir
    # work_dir = '/workspace/host'
    work_dir = '/home/exx/Documents/NAS/Datasets/alanweiyang'

    # inputs from dataset or test random tensors, if dataset is wanted, then need to get into docker alan_imuv by docker exec -it alan_imuv bash, otherwise comment below two lines for testing
    # from dataset import LibriMixIMUV
    # dataset = LibriMixIMUV(
    #     csv_dir='/workspace/host/LibriMix/dataset/Libri3Mix/wav16k/min/metadata',
    #     # csv_dir='/home/exx/Documents/NAS/Datasets/alanweiyang/LibriMix/dataset/Libri3Mix/wav16k/min/metadata',
    #     sample_rate=16000, 
    #     noisy=True,
    #     min_num_sources=1,
    #     max_num_sources=3,
    #     subband_snr_range=[3, 10],
    #     # segment=5,
    #     mode='dev'     
    # )
    # batch = dataset[2]
    # mixture = batch['mixture']
    # target = batch['target']
    # input = batch['mixture'].unsqueeze(0)
    # subband_clean_input = batch['target_sb_noisy'].unsqueeze(0)

    # if the dataset is available then just testing on random data
    input = torch.randn(1, 16000)
    subband_clean_input = torch.randn(1, 16000)
    mixture = torch.randn(16000)
    target = torch.randn(16000)

    # checkpoint dir
    ckpt_dir = os.path.join(work_dir, 'alan_imuv/models/ckpt_imu_tasnet')

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

    torch.set_printoptions(precision=10)

    # set device
    model = model.cuda()
    mixture = mixture.cuda()
    input = input.cuda()
    subband_clean_input = subband_clean_input.cuda()
    target = target.cuda()


    # running inference
    with torch.no_grad():
        output = model.inference(input, subband_clean_input) # 1, n_samples


    # currently the output is scale invariant....need to scale to the same level as the target, but we don't know the target
    output = output.squeeze(0)
    alpha_mix = (output * mixture).sum() / (output**2).sum().clamp_min(1e-8)
    output = alpha_mix * output # this is the final output, (1, n_samples)

    # sanity check the snr
    print('mixture snr: ', 10 * torch.log10(target.square().sum() / (mixture-target).square().sum()))
    print('output snr', 10 * torch.log10(target.square().sum() / (output-target).square().sum()))



if __name__ == "__main__":
    test()