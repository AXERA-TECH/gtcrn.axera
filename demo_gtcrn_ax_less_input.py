import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from axengine import InferenceSession
import torch
import time


def init_caches():
    return {
        'en_conv_cache': np.zeros([1, 16, 16, 33], dtype="float32"),
        'de_conv_cache': np.zeros([1, 16, 16, 33], dtype="float32"),
        'en_tra_cache': np.zeros([1, 3, 1, 16], dtype="float32"),
        'de_tra_cache': np.zeros([1, 3, 1, 16], dtype="float32"),
        'inter_cache_0': np.zeros([1, 1, 33, 16], dtype="float32"),
        'inter_cache_1': np.zeros([1, 1, 33, 16], dtype="float32"),
    }


def update_caches(outputs):
    # outputs order: enh, en_conv_cache_out, de_conv_cache_out,
    # en_tra_cache_out, de_tra_cache_out, inter_cache_0_out, inter_cache_1_out
    return {
        'en_conv_cache': outputs[1],
        'de_conv_cache': outputs[2],
        'en_tra_cache': outputs[3],
        'de_tra_cache': outputs[4],
        'inter_cache_0': outputs[5],
        'inter_cache_1': outputs[6],
    }


def denoise_audio(model_path, input_audio_path, output_audio_path,
                  n_fft=512, hop_length=256, sample_rate=16000):
    print(f">>> 加载模型: {model_path}")
    session = InferenceSession(model_path)
    print(f">>> 读取音频: {input_audio_path}")
    start_time = time.time()
    audio, sr = sf.read(input_audio_path, dtype='float32')
    if sr != sample_rate:
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=sample_rate).T
        sr = sample_rate
    assert sr == sample_rate, f"采样率不匹配: {sr} != {sample_rate}"
    print(f">>> 音频长度: {len(audio)/sr:.2f}秒")
    print(">>> 执行 STFT...")
    stft_start = time.time()
    audio_tensor = torch.from_numpy(audio)
    window = torch.hann_window(n_fft).pow(0.5)
    spec = torch.stft(audio_tensor, n_fft, hop_length, n_fft, window,
                      return_complex=False)
    spec = spec.numpy()[np.newaxis, ...]
    stft_end = time.time()
    print(f"STFT耗时: {stft_end - stft_start:.4f} 秒")
    print(f">>> 频谱 shape: {spec.shape}")

    cache_dict = init_caches()
    print(">>> model infer...")
    enhanced_frames = []
    model_infer_time = 0.0
    for i in tqdm(range(spec.shape[2]), desc="处理中"):
        frame_spec = spec[:, :, i:i+1, :]
        input_dict = {'mix': frame_spec}
        input_dict.update(cache_dict)
        infer_start = time.time()
        outputs = session.run(None, input_dict)
        infer_end = time.time()
        model_infer_time += (infer_end - infer_start)
        enhanced_frame = outputs[0]
        enhanced_frames.append(enhanced_frame)
        cache_dict = update_caches(outputs)

    enhanced_spec = np.concatenate(enhanced_frames, axis=2)
    print(">>> 执行 ISTFT...")
    istft_start = time.time()
    enhanced_spec_complex = enhanced_spec[0, :, :, 0] + 1j * enhanced_spec[0, :, :, 1]

    real = torch.from_numpy(enhanced_spec_complex.real).unsqueeze(0).contiguous()
    imag = torch.from_numpy(enhanced_spec_complex.imag).unsqueeze(0).contiguous()
    stft_tensor = torch.complex(real, imag)  # (1, freq, time)

    window_np = np.hanning(n_fft) ** 0.5
    window = torch.from_numpy(window_np).float()

    enhanced_audio_tensor = torch.istft(
        stft_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        length=None
    )
    enhanced_audio = enhanced_audio_tensor.squeeze(0).numpy().astype(np.float32)

    istft_end = time.time()
    all_time = time.time() - start_time
    print(f"ISTFT耗时: {istft_end - istft_start:.4f} 秒")
    print(f"模型推理耗时: {model_infer_time:.4f} 秒")
    print(f"总耗时: {all_time:.4f} 秒")
    audio_duration = len(audio) / sample_rate
    rtf = all_time / audio_duration if audio_duration > 0 else float('inf')
    print(f"音频时长: {audio_duration:.2f} 秒")
    print(f"RTF: {rtf:.4f}")
    print(f"保存降噪音频: {output_audio_path}")
    sf.write(output_audio_path, enhanced_audio, sample_rate)
    print(">>> 完成!")
    return enhanced_audio


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GTCRN AX demo (taishan)')
    parser.add_argument('--model', type=str,
                       default='models/gtcrn.axmodel',
                       help='模型路径')
    parser.add_argument('--input', type=str,
                       default='test_wavs/mix.wav',
                       help='输入音频路径')
    parser.add_argument('--output', type=str,
                       default='test_wavs/demo_output_ax630_taishan.wav',
                       help='输出音频路径')
    args = parser.parse_args()

    denoise_audio(args.model, args.input, args.output)
