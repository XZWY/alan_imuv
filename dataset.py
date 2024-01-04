import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import hub
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import shutil
import zipfile
import librosa

class LibriMix(Dataset):
    """Dataset class for LibriMix source separation tasks.

    Args:
        csv_dir (str): The path to the metadata file.
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'`` :

            * ``'enh_single'`` for single speaker speech enhancement.
            * ``'enh_both'`` for multi speaker speech enhancement.
            * ``'sep_clean'`` for two-speaker clean source separation.
            * ``'sep_noisy'`` for two-speaker noisy source separation.

        sample_rate (int) : The sample rate of the sources and mixtures.
        n_src (int) : The number of sources in the mixture.
        segment (int, optional) : The desired sources and mixtures length in s.

    References
        [1] "LibriMix: An Open-Source Dataset for Generalizable Speech Separation",
        Cosentino et al. 2020.
    """

    dataset_name = "LibriMix"

    def __init__(
        self, csv_dir, task="sep_clean", sample_rate=16000, n_src=2, segment=3, return_id=False
    ):
        self.csv_dir = csv_dir
        self.task = task
        self.return_id = return_id
        # Get the csv corresponding to the task
        if task == "enh_single":
            md_file = [f for f in os.listdir(csv_dir) if ("single" in f and '360' in f and 'mixture' in f)][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        elif task == "enh_both":
            md_file = [f for f in os.listdir(csv_dir) if ("both" in f and '360' in f and 'mixture' in f)][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
            md_clean_file = [f for f in os.listdir(csv_dir) if ("clean" in f and '360' in f and 'mixture' in f)][0]
            self.df_clean = pd.read_csv(os.path.join(csv_dir, md_clean_file))
        elif task == "sep_clean":
            md_file = [f for f in os.listdir(csv_dir) if ("clean" in f and '360' in f and 'mixture' in f)][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        elif task == "sep_noisy":
            md_file = [f for f in os.listdir(csv_dir) if ("both" in f and '360' in f and 'mixture' in f)][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        self.segment = segment
        self.sample_rate = sample_rate
        # Open csv file
        self.df = pd.read_csv(self.csv_path)
        # Get rid of the utterances too short
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            # Ignore the file shorter than the desired_length
            self.df = self.df[self.df["length"] >= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None
        self.n_src = n_src

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.df.iloc[idx]
        # Get mixture path
        mixture_path = row["mixture_path"]
        self.mixture_path = mixture_path
        sources_list = []
        # If there is a seg start point is set randomly
        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None
        # If task is enh_both then the source is the clean mixture
        if "enh_both" in self.task:
            mix_clean_path = self.df_clean.iloc[idx]["mixture_path"]
            s, _ = sf.read(mix_clean_path, dtype="float32", start=start, stop=stop)
            sources_list.append(s)

        else:
            # Read sources
            for i in range(self.n_src):
                source_path = row[f"source_{i + 1}_path"]
                s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
                sources_list.append(s)
        
        # load noise if noisy
        if "clean" not in self.task:
            noise_path = self.df.iloc[idx]["noise_path"]
            noise, _ = sf.read(noise_path, dtype="float32", start=start, stop=stop)
        # Read the mixture
        mixture, _ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
        # Convert to torch tensor
        mixture = torch.from_numpy(mixture)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        
        # if not self.return_id:
        #     return mixture, sources
        # 5400-34479-0005_4973-24515-0007.wav
        # print(mixture_path.split("/")[-1].split(".")[0].split("_"))
        # id1, id2 = mixture_path.split("/")[-1].split(".")[0].split("_")
        
        batch = {}
        batch['mixture'] = mixture
        batch['sources'] = sources
        if "clean" not in self.task:
            noise = torch.from_numpy(noise)
            batch['noise'] = noise
        # batch['ids'] = [id1, id2]
        return batch

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self._dataset_name()
        infos["task"] = self.task
        return infos

    def _dataset_name(self):
        """Differentiate between 2 and 3 sources."""
        return f"Libri{self.n_src}Mix"
    

class LibriMixIMUV(Dataset):
    """Dataset class for LibriMix source separation tasks.

    Args:
        csv_dir (str): The path to the metadata file.
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'`` :

            * ``'enh_single'`` for single speaker speech enhancement.
            * ``'enh_both'`` for multi speaker speech enhancement.
            * ``'sep_clean'`` for two-speaker clean source separation.
            * ``'sep_noisy'`` for two-speaker noisy source separation.

        sample_rate (int) : The sample rate of the sources and mixtures.
        n_src (int) : The number of sources in the mixture.
        segment (int, optional) : The desired sources and mixtures length in s.

    References
        [1] "LibriMix: An Open-Source Dataset for Generalizable Speech Separation",
        Cosentino et al. 2020.
    """

    dataset_name = "LibriMixIMUV"

    def __init__(
        self,
        csv_dir,
        sample_rate=16000,
        noisy=True,
        min_num_sources=1,
        max_num_sources=3,
        subband_snr_range=[3, 10],
        segment=5,
    ):
        self.csv_dir = csv_dir
        self.noisy = noisy
        self.min_num_sources = min_num_sources
        self.max_num_sources = max_num_sources
        self.subband_snr_range = subband_snr_range

        # Get the csv corresponding to the task, default to be libri3mix, noisy
        md_file = [f for f in os.listdir(csv_dir) if ("both" in f and '360' in f and 'mixture' in f)][0]
        self.csv_path = os.path.join(self.csv_dir, md_file)
        md_clean_file = [f for f in os.listdir(csv_dir) if ("clean" in f and '360' in f and 'mixture' in f)][0]
        self.df_clean = pd.read_csv(os.path.join(csv_dir, md_clean_file))

        self.segment = segment
        self.sample_rate = sample_rate
        # Open csv file
        self.df = pd.read_csv(self.csv_path)
        # Get rid of the utterances too short
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            # Ignore the file shorter than the desired_length
            self.df = self.df[self.df["length"] >= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.df.iloc[idx]
        # Get mixture path
        mixture_path = row["mixture_path"]
        self.mixture_path = mixture_path
        sources_list = []
        # If there is a seg start point is set randomly
        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None
        # If task is enh_both then the source is the clean mixture

        # Read sources
        n_src = np.random.randint(self.min_num_sources, self.max_num_sources+1)
        shuffle_indices = (np.arange(0, 3))
        np.random.shuffle(shuffle_indices)
        print(n_src)
        for i in range(n_src):
            source_path = row[f"source_{shuffle_indices[i] + 1}_path"]
            s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
            sources_list.append(s)
        
        # load noise if noisy
        if self.noisy:
            noise_path = self.df.iloc[idx]["noise_path"]
            noise, _ = sf.read(noise_path, dtype="float32", start=start, stop=stop)
        # # Read the mixture
        # mixture, _ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
        # # Convert to torch tensor
        # mixture = torch.from_numpy(mixture)
        # Stack sources
        sources = np.vstack(sources_list)
        
        tgt_idx = np.random.randint(0, n_src)
        target = sources[tgt_idx]
        
        # create noisy mixture        
        mixture = sources.sum(0)
        if self.noisy:
            mixture = mixture + noise
        
        # get subband feature
        subband_target_snr = np.random.uniform(self.subband_snr_range[0], self.subband_snr_range[1])
        noise_interference = mixture - target
        target_sb = librosa.resample(librosa.resample(target, 16000, 2000), 2000, 16000)
        noise_interference_sb = librosa.resample(librosa.resample(noise_interference, 16000, 2000), 2000, 16000)
        alpha = np.linalg.norm(target_sb) / (np.linalg.norm(noise_interference_sb) * 10**(subband_target_snr / 20))
        alpha = np.clip(alpha, 0, 0.9)
        noisy_target_sb = target_sb + alpha * noise_interference_sb
        
        # print(alpha, subband_target_snr)
        snr = 20 * np.log10(np.linalg.norm(target_sb) / (np.linalg.norm(noisy_target_sb-target_sb) + 1e-8))
        
        batch = {}
        batch['snr'] = torch.tensor(snr).type(torch.float32)
        # Convert sources to tensor
        mixture = torch.from_numpy(mixture)
        target = torch.from_numpy(target)
        target_sb = torch.from_numpy(target_sb)
        target_sb_noisy = torch.from_numpy(noisy_target_sb)
        
        batch['mixture'] = mixture
        batch['target'] = target
        batch['target_sb'] = target_sb
        batch['target_sb_noisy'] = target_sb_noisy
        
        return batch

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self._dataset_name()
        infos["task"] = self.task
        return infos

    def _dataset_name(self):
        """Differentiate between 2 and 3 sources."""
        return f"Libri{self.n_src}Mix"


def collate_func_separation(batches):
    '''
        collate function for general speech separation and enhancement
    '''
    new_batch = {}
    for key in batches[0].keys():
        new_batch[key] = [] 
    for batch in batches:
        # print(key, len(new_batch[key]))
        for key in new_batch.keys():
            new_batch[key].append(batch[key])
    for key in new_batch.keys():
        new_batch[key] = torch.stack(new_batch[key], dim=0)
    
    return new_batch
    
if __name__=='__main__':
    
    # * ``'enh_single'`` for single speaker speech enhancement.
    # * ``'enh_both'`` for multi speaker speech enhancement.
    # * ``'sep_clean'`` for two-speaker clean source separation.
    # * ``'sep_noisy'`` for two-speaker noisy source separation.
    
    dataset = LibriMix(csv_dir='/workspace/host/LibriMix/dataset/Libri3Mix/wav16k/min/metadata', task="sep_noisy", sample_rate=16000, n_src=3, segment=5, return_id=False)
    # batch = collate_func_separation([dataset[0], dataset[1]])
    # # print(batch)
    # for key in batch.keys():
    #     if type(batch[key]) is torch.Tensor:
    #         print(key, batch[key].shape)
    
    batch = dataset[0]
    for key in batch.keys():
        if type(batch[key]) is torch.Tensor:
            print(key, batch[key].shape)
            
    new_mixture = batch['sources'].sum(0) + batch['noise']
    # sf.write('mixture.wav', batch['mixture'], 16000)
    # sf.write('new_mixture.wav', new_mixture, 16000)
    
    dataset = LibriMixIMUV(
        csv_dir='/workspace/host/LibriMix/dataset/Libri3Mix/wav16k/min/metadata',
        sample_rate=16000,
        noisy=True,
        min_num_sources=1,
        max_num_sources=3,
        subband_snr_range=[3, 10],
        segment=5        
    )        
    # batch = collate_func_separation([dataset[0], dataset[1]])
    # # print(batch)
    # for key in batch.keys():
    #     if type(batch[key]) is torch.Tensor:
    #         print(key, batch[key].shape)
    batch = dataset[0]
    for key in batch.keys():
        if type(batch[key]) is torch.Tensor:
            print(key, batch[key].shape)
    
    # batch = collate_func_separation([dataset[0], dataset[1]])
    # # print(batch)
    # print(batch['snr'])
    # for key in batch.keys():
    #     if type(batch[key]) is torch.Tensor:
    #         print(key, batch[key].shape)
    sf.write('target.wav', batch['target'], 16000)
    sf.write('target_sb.wav', batch['target_sb'], 16000)
    sf.write('target_sb_noisy.wav', batch['target_sb_noisy'], 16000)