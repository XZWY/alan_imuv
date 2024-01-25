import torch.nn as nn
import torch.nn.functional as F
import torch

class Suband_Encoder(nn.Module):
     """
     Input: [batch size, channels=1, T, n_fft]
     Output: [batch size, T, n_fft]
     """
     def __init__(self):
          super(Suband_Encoder, self).__init__()
          # Encoder
          self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3, 3), stride=(1, 2))
          self.bn1 = nn.BatchNorm2d(num_features=16)
          self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
          self.bn2 = nn.BatchNorm2d(num_features=32)
          self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 2))
          self.bn3 = nn.BatchNorm2d(num_features=64)
          self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 2))
          self.bn4 = nn.BatchNorm2d(num_features=128)
          self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 2))
          self.bn5 = nn.BatchNorm2d(num_features=128)
     def forward(self, x):
          x1 = F.elu(self.bn1(self.conv1(F.pad(x, (0,0,1,1)))))
          x2 = F.elu(self.bn2(self.conv2(F.pad(x1, (0,0,1,1)))))
          x3 = F.elu(self.bn3(self.conv3(F.pad(x2, (0,0,1,1)))))
          x4 = F.elu(self.bn4(self.conv4(F.pad(x3, (0,0,1,1)))))
          x5 = F.elu(self.bn5(self.conv5(F.pad(x4, (0,0,1,1)))))
          # reshape
          out5 = x5.permute(0, 2, 1, 3)
          out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
          return out5

class CRNN(nn.Module):
     """
     Input: [batch size, channels=1, T, n_fft]
     Output: [batch size, T, n_fft]
     """
     def __init__(self, input_channel=2):
          super(CRNN, self).__init__()

          self.num_speakers = 1

          self.subband_encoder = Suband_Encoder()
          
          # Encoder
          self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=(3, 3), stride=(1, 2))
          self.bn1 = nn.BatchNorm2d(num_features=16)
          self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
          self.bn2 = nn.BatchNorm2d(num_features=32)
          self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 2))
          self.bn3 = nn.BatchNorm2d(num_features=64)
          self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 2))
          self.bn4 = nn.BatchNorm2d(num_features=128)
          self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 2))
          self.bn5 = nn.BatchNorm2d(num_features=128)

          # LSTM
          self.LSTM1 = nn.LSTM(input_size=1024, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)

          # Decoder
          self.convT1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 2))
          self.bnT1 = nn.BatchNorm2d(num_features=128)
          self.convT2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(3, 3), stride=(1, 2))
          self.bnT2 = nn.BatchNorm2d(num_features=64)
          self.convT3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
          self.bnT3 = nn.BatchNorm2d(num_features=32)
          # output_padding为1，不然算出来是79
          self.convT4 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(3, 3), stride=(1, 2), output_padding=(0, 1))
          self.bnT4 = nn.BatchNorm2d(num_features=16)
          self.convT5 = nn.ConvTranspose2d(in_channels=32, out_channels=input_channel*self.num_speakers, kernel_size=(3, 3), stride=(1, 2))
          self.bnT5 = nn.BatchNorm2d(num_features=input_channel*self.num_speakers)


     def stft(self, input):
          return torch.stft(input, n_fft=320, hop_length=160, win_length=320, window=torch.hann_window(320).to(input.device), return_complex=True)
     
     def istft(self, input):
          return torch.istft(input, n_fft=320, hop_length=160, win_length=320, window=torch.hann_window(320).to(input.device))

     def forward(self, input, conditional_input):
          # conv
          # 
          input_length = input.shape[1]
          x = torch.view_as_real(self.stft(F.pad(input, (0, 320)))).permute(0,3,2,1) # (B, in_c, T, F)
          sub_x = torch.view_as_real(self.stft(F.pad(conditional_input, (0, 320)))).permute(0,3,2,1) # (B, in_c, T, F)
          bs, _, n_frames, n_freqs = x.shape
          sub_enc_out = self.subband_encoder(sub_x)

          x1 = F.elu(self.bn1(self.conv1(F.pad(x, (0,0,1,1)))))
          x2 = F.elu(self.bn2(self.conv2(F.pad(x1, (0,0,1,1)))))
          x3 = F.elu(self.bn3(self.conv3(F.pad(x2, (0,0,1,1)))))
          x4 = F.elu(self.bn4(self.conv4(F.pad(x3, (0,0,1,1)))))
          x5 = F.elu(self.bn5(self.conv5(F.pad(x4, (0,0,1,1)))))
          # reshape
          out5 = x5.permute(0, 2, 1, 3)
          out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
          # lstm

          lstm, (hn, cn) = self.LSTM1(torch.cat([out5, sub_enc_out], dim=2))
          # reshape
          output = lstm.reshape(lstm.size()[0], lstm.size()[1], 128, -1) # bs, n_frames, 256, 4
          output = output.permute(0, 2, 1, 3) # bs, 256, n_frames, 4
          # ConvTrans
          res = torch.cat((output, x5), 1)
          res1 = F.elu(self.bnT1(self.convT1(res)))
          res1 = torch.cat((res1[:, :, 1:-1], x4), 1)
          res2 = F.elu(self.bnT2(self.convT2(res1)))
          res2 = torch.cat((res2[:, :, 1:-1], x3), 1)
          res3 = F.elu(self.bnT3(self.convT3(res2)))
          res3 = torch.cat((res3[:, :, 1:-1], x2), 1)
          res4 = F.elu(self.bnT4(self.convT4(res3)))
          res4 = torch.cat((res4[:, :, 1:-1], x1), 1)
          # (B, o_c, T. F)
          res5 = F.relu(self.bnT5(self.convT5(res4[:, :, 1:-1]))) # bs, 2, T, F

          out = torch.view_as_complex(res5.permute(0,3,2,1).contiguous()) * torch.view_as_complex(x.permute(0,3,2,1).contiguous()) # bs, T, F
          # print(x.shape, res5.shape)
          out = self.istft(out)[..., :input_length]
          return out


if __name__=='__main__':
     model = CRNN(input_channel=2)

     input = torch.rand(2, 16000)
     output = model(input, input)

     print(output.shape)

