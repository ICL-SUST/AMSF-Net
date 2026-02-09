import numpy as np
import torch
import pywt
from PIL import Image
import cv2


class MultiScaleWFDTTransform:

    def __init__(self, wavelet='db4', levels=3):
        self.wavelet = wavelet
        self.levels = levels

    def __call__(self, img):

        if img.mode == 'RGB':
            img = img.convert('L')

        img_array = np.array(img).astype(np.float32)
        original_size = img_array.shape

        img_normalized = img_array / 255.0

        all_bands = []

        coeffs = img_normalized
        for level in range(1, self.levels + 1):
            coeffs_2d = pywt.dwt2(coeffs, self.wavelet)
            LL, (LH, HL, HH) = coeffs_2d

            if level == 1:
                LL_recon = self._idwt_single_band(LL, original_size, level, 'LL')
                all_bands.append(LL_recon)

            LH_recon = self._idwt_single_band(LH, original_size, level, 'LH')
            HL_recon = self._idwt_single_band(HL, original_size, level, 'HL')
            HH_recon = self._idwt_single_band(HH, original_size, level, 'HH')

            LH_norm = self._normalize_band(LH_recon)
            HL_norm = self._normalize_band(HL_recon)
            HH_norm = self._normalize_band(HH_recon)

            all_bands.extend([LH_norm, HL_norm, HH_norm])

            coeffs = LL

        multi_channel_output = np.stack(all_bands, axis=0)

        return multi_channel_output

    def _idwt_single_band(self, band, target_size, level, band_type):

        current = band

        for l in range(level, 0, -1):
            if l == 1:
                h, w = target_size
            else:
                h = current.shape[0] * 2
                w = current.shape[1] * 2
            if band_type == 'LL':

                coeffs_recon = (current, (np.zeros_like(current),
                                          np.zeros_like(current),
                                          np.zeros_like(current)))
            elif band_type == 'LH':
                coeffs_recon = (np.zeros_like(current),
                                (current, np.zeros_like(current), np.zeros_like(current)))
            elif band_type == 'HL':
                coeffs_recon = (np.zeros_like(current),
                                (np.zeros_like(current), current, np.zeros_like(current)))
            else:
                coeffs_recon = (np.zeros_like(current),
                                (np.zeros_like(current), np.zeros_like(current), current))

            current = pywt.idwt2(coeffs_recon, self.wavelet)

            current = current[:h, :w]

        if current.shape != target_size:
            current = cv2.resize(current, (target_size[1], target_size[0]),
                                 interpolation=cv2.INTER_LINEAR)

        return current

    def _normalize_band(self, band):

        p5 = np.percentile(band, 5)
        p95 = np.percentile(band, 95)

        if p95 > p5:
            band_norm = (band - p5) / (p95 - p5 + 1e-8)
            band_norm = np.clip(band_norm, 0, 1)
        else:
            band_norm = np.zeros_like(band)

        gamma = 0.7
        band_norm = np.power(band_norm + 1e-8, gamma)

        band_norm = band_norm * 255

        return band_norm.astype(np.float32)


class WFDTTransform:

    def __init__(self, wavelet='haar', level=1):
        self.wavelet = wavelet
        self.level = level

    def __call__(self, img):
        if img.mode == 'RGB':
            img = img.convert('L')

        high_freq_img = self._process_channel(img)
        return high_freq_img

    def _process_channel(self, channel_img):
        img_array = np.array(channel_img).astype(np.float32)

        coeffs = pywt.dwt2(img_array, self.wavelet)
        LL, (LH, HL, HH) = coeffs

        high_freq_combined = HL + LH + HH

        mean = np.mean(high_freq_combined)
        std = np.std(high_freq_combined) + 1e-8
        high_freq_norm = (high_freq_combined - mean) / std

        high_freq_norm = np.tanh(high_freq_norm / 2)
        high_freq_norm = (high_freq_norm + 1) * 127.5

        high_freq_resized = cv2.resize(
            high_freq_norm,
            (channel_img.size[0], channel_img.size[1]),
            interpolation=cv2.INTER_LINEAR
        )

        high_freq_resized = np.clip(high_freq_resized, 0, 255)

        high_freq_channel = Image.fromarray(high_freq_resized.astype(np.uint8), mode='L')

        return high_freq_channel