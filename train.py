# from hificodec train.py @ https://github.com/yangdongchao/AcademiCodec
# pipeline for training rnnbf-hificodec
# weiyang 2023-05-26

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

import os
import shutil

try:
    from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint, mel_spectrogram
except:
    from .utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint, mel_spectrogram

from dataset import LibriMix, LibriMixIMUV, collate_func_separation
from models.imuv_tasnet_sisnr import IMUV_TASNET_SISNR

torch.backends.cudnn.benchmark = True


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))



def train(rank, a, h):
    if h.num_gpus > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12498'
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    model = IMUV_TASNET_SISNR().to(device)

    if rank == 0:
        # print(rnnbf)
        print(model)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        ckpt = scan_checkpoint(a.checkpoint_path, 'ckpt_')

    steps = 0
    # last_epoch = -1
    if ckpt is None:
        state_dict = None
        last_epoch = -1
    else:
        state_dict = load_checkpoint(ckpt, device)
        model.load_state_dict(state_dict['model'])
        steps = state_dict['steps'] + 1
        last_epoch = state_dict['epoch']
        print('load ckpt successfully....................................')

    if h.num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True).to(device)


    optim = torch.optim.Adam(itertools.chain(model.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])


    if ckpt is not None:
        optim.load_state_dict(state_dict['optim'])
        optim.param_groups[0]['capturable'] = True

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=h.lr_decay, last_epoch=last_epoch)

    trainset = LibriMixIMUV(
        csv_dir='/workspace/host/LibriMix/dataset/Libri3Mix/wav16k/min/metadata',
        sample_rate=16000,
        noisy=True,
        min_num_sources=1,
        max_num_sources=3,
        subband_snr_range=[3, 10],
        segment=5,
        mode='train'
        )

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=collate_func_separation)

    if rank == 0:
        validset = LibriMixIMUV(
            csv_dir='/workspace/host/LibriMix/dataset/Libri3Mix/wav16k/min/metadata',
            sample_rate=16000,
            noisy=True,
            min_num_sources=1,
            max_num_sources=3,
            subband_snr_range=[3, 10],
            mode='dev'
        )
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                        sampler=None,
                                        batch_size=1,
                                        pin_memory=True,
                                        drop_last=True,
                                        collate_fn=collate_func_separation)
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
    model.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            # if i >= 30:
            #     break
            if rank == 0:
                start_b = time.time()

            for key in batch.keys():
                if type(batch[key]) is torch.Tensor:
                    batch[key].to(device)

            # model forward and loss calculation
            optim.zero_grad()
            batch = model(batch)
            loss = batch['loss'].mean()
            
            loss.backward()

            optim.step()
            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                # if steps % 5 == 0:

                    print('Steps : {:d}, SISNR lOSS : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss, time.time() - start_b))
                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                # if steps % 20 == 0 and steps != 0:
                    checkpoint_path = "{}/ckpt_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'model': (model.module if h.num_gpus > 1 else model).state_dict(),
                                     'optim': optim.state_dict(), 'steps': steps,
                                     'epoch': epoch}, num_ckpt_keep=a.num_ckpt_keep)            

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/sisnr_loss", loss, steps)


                # Validation
                if steps % a.validation_interval == 0 and steps != 0:
                # if steps % 20 == 0 and steps != 0:
                    model.eval()
                    torch.cuda.empty_cache()
                    val_snr_tot = 0

                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            # if j > 10:
                            #     break
                            # -------------------------------------
                            for key in batch.keys():
                                if type(batch[key]) is torch.Tensor:
                                    batch[key].to(device)

                            # model forward and loss calculation
                            batch = model(batch)
                            

                            print('val snr: ', -batch['loss'])
                            val_snr_tot += (-batch['loss'].mean().item())

                            if j <= 8:
                                # if steps == 0:
                                
                                sw.add_audio('mixture/y_{}'.format(j), batch['mixture'][0], steps, h.sampling_rate)
                                y_spec = mel_spectrogram(batch['mixture'][0].unsqueeze(0), h.n_fft, h.num_mels,
                                                            h.sampling_rate, h.hop_size, h.win_size,
                                                            h.fmin, h.fmax)
                                sw.add_figure('mixture/y_spec_{}'.format(j),
                                            plot_spectrogram(y_spec.squeeze(0).cpu().numpy()), steps)

                                sw.add_audio('prompt/y_{}'.format(j), batch['target_sb_noisy'][0], steps, h.sampling_rate)
                                y_spec = mel_spectrogram(batch['target_sb_noisy'][0].unsqueeze(0), h.n_fft, h.num_mels,
                                                            h.sampling_rate, h.hop_size, h.win_size,
                                                            h.fmin, h.fmax)
                                sw.add_figure('prompt/y_spec_{}'.format(j),
                                            plot_spectrogram(y_spec.squeeze(0).cpu().numpy()), steps)
                                
                                sw.add_audio('output/y_{}'.format(j), batch['output'][0], steps, h.sampling_rate)
                                y_spec = mel_spectrogram(batch['output'][0].unsqueeze(0), h.n_fft, h.num_mels,
                                                            h.sampling_rate, h.hop_size, h.win_size,
                                                            h.fmin, h.fmax)
                                sw.add_figure('output/y_spec_{}'.format(j),
                                            plot_spectrogram(y_spec.squeeze(0).cpu().numpy()), steps)
                                

                        val_snr_tot = val_snr_tot / (j+1)
                        sw.add_scalar("validation/snr", val_snr_tot, steps)

                        model.train()
                    
            steps += 1

        scheduler.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    # parser.add_argument('--group_name', default=None)
    # parser.add_argument('--input_wavs_dir', default='../datasets/audios')
    parser.add_argument('--input_mels_dir', default=None)
    # parser.add_argument('--in_path', required=True)
    # parser.add_argument('--scp_path', required=True)
    parser.add_argument('--checkpoint_path', default='checkpoints')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument('--num_ckpt_keep', default=5, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()