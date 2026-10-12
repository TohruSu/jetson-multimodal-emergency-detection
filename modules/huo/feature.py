import torch
import torch.nn.functional as F


# -----------------------------
# STFT + Mel filter (pure torch)
# -----------------------------

class AudioFeatureExtractor:
    def __init__(self,
                 sample_rate=16000,
                 n_fft=400,
                 hop_length=160,
                 n_mels=64,
                 f_min=50,
                 f_max=8000):

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # hann window
        self.window = torch.hann_window(n_fft)

        # mel filter bank
        self.mel_fb = self._create_mel_filter(n_fft, n_mels, f_min, f_max)

    def _hz_to_mel(self, hz):
        return 2595 * torch.log10(1 + hz / 700)

    def _mel_to_hz(self, mel):
        return 700 * (10**(mel / 2595) - 1)

    def _create_mel_filter(self, n_fft, n_mels, f_min, f_max):
        n_freqs = n_fft // 2 + 1

        mels = torch.linspace(
            self._hz_to_mel(torch.tensor(f_min)),
            self._hz_to_mel(torch.tensor(f_max)),
            n_mels + 2
        )

        hz = self._mel_to_hz(mels)

        bins = torch.floor((n_fft + 1) * hz / self.sample_rate).long()

        fb = torch.zeros(n_mels, n_freqs)

        for i in range(1, n_mels + 1):
            left = bins[i - 1]
            center = bins[i]
            right = bins[i + 1]

            for j in range(left, center):
                if 0 <= j < n_freqs:
                    fb[i - 1, j] = (j - left) / (center - left + 1e-6)

            for j in range(center, right):
                if 0 <= j < n_freqs:
                    fb[i - 1, j] = (right - j) / (right - center + 1e-6)

        return fb

    def extract(self, wav):
        """
        wav: Tensor [T]
        return: [1, 64, time]
        """

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        #if wav.ndim == 1:
        #    wav = np.expand_dims(wav,0)

        # STFT
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )

        spec = torch.abs(spec) ** 2  # power spectrogram

        # Mel filter
        mel = torch.matmul(self.mel_fb, spec.squeeze(0))

        # log scale
        mel = torch.log(mel + 1e-6)

        # normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # resize to 224x224 for ResNet
        #mel = mel.unsqueeze(0).unsqueeze(0)  # [1,1,mels,time]
        mel = mel.unsqueeze(0)  # [1,freq,time]
        mel = F.interpolate(mel.unsqueeze(0), size=(224, 224), mode="bilinear",align_corners=False).squeeze(0)

        return mel
