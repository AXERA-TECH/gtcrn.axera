import os
import torch
import numpy as np
import soundfile as sf
import argparse
import zipfile
from tqdm import tqdm
from pathlib import Path
import shutil
import onnxruntime


def prepare_output_dirs(output_dir):
    output_dir = Path(output_dir)
    
    temp_dir = output_dir / "temp_npy"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    zip_dir = output_dir
    zip_dir.mkdir(parents=True, exist_ok=True)
    
    return temp_dir, zip_dir


def collect_audio_files(audio_dir, max_files=None):
    audio_dir = Path(audio_dir)
    audio_files = []
    
    for ext in ['*.wav', '*.flac', '*.mp3']:
        audio_files.extend(audio_dir.glob(ext))
    
    audio_files = sorted(audio_files)
    
    if max_files:
        audio_files = audio_files[:max_files]
    
    print(f">>> 找到 {len(audio_files)} 个音频文件")
    return audio_files


def audio_to_spec(audio_path):
    audio, sr = sf.read(str(audio_path), dtype='float32')
    
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    if sr != 16000:
        print(f"\n警告: {audio_path.name} 采样率为 {sr}，期望 16000")

    audio_tensor = torch.from_numpy(audio)
    spec = torch.stft(
        audio_tensor, 
        n_fft=512, 
        hop_length=256, 
        win_length=512, 
        window=torch.hann_window(512).pow(0.5), 
        return_complex=False
    )
    
    spec = spec.unsqueeze(0).numpy()
    
    return spec


def init_caches():
    return {
        'en_conv_cache': np.zeros([1, 16, 16, 33], dtype="float32"),
        'de_conv_cache': np.zeros([1, 16, 16, 33], dtype="float32"),
        'en_tra_cache': np.zeros([1, 3, 1, 16], dtype="float32"),
        'de_tra_cache': np.zeros([1, 3, 1, 16], dtype="float32"),
        'inter_cache_0': np.zeros([1, 1, 33, 16], dtype="float32"),
        'inter_cache_1': np.zeros([1, 1, 33, 16], dtype="float32"),
    }


def generate_calibration_data_with_onnx(onnx_model_path, audio_files, num_samples, 
                                         temp_dir, skip_frames=1, warmup_frames=10):
    
    input_names = [
        'mix',
        'en_conv_cache', 'de_conv_cache',
        'en_tra_cache', 'de_tra_cache',
        'inter_cache_0', 'inter_cache_1'
    ]
    
    input_dirs = {}
    for name in input_names:
        input_dir = temp_dir / name
        input_dir.mkdir(exist_ok=True)
        input_dirs[name] = input_dir
    
    print(f"\n>>> 加载 ONNX 模型: {onnx_model_path}")
    session = onnxruntime.InferenceSession(
        onnx_model_path, 
        None,
        providers=['CPUExecutionProvider']
    )
    
    sample_count = 0
    
    print(f"\n>>> 开始生成校准数据，目标样本数: {num_samples}")
    print(f">>> 跳帧间隔: {skip_frames} (warmup: {warmup_frames} 帧)")
    
    pbar = tqdm(total=num_samples, desc="生成样本")
    
    for audio_idx, audio_file in enumerate(audio_files):
        if sample_count >= num_samples:
            break
        
        try:
            spec = audio_to_spec(audio_file)
            num_frames = spec.shape[2]

            caches = init_caches()
            
            warmup_end = min(warmup_frames, num_frames)
            for frame_idx in range(warmup_end):
                frame = spec[:, :, frame_idx:frame_idx+1, :]
                input_dict = {'mix': frame}
                input_dict.update(caches)
                
                # 推理
                output_list = session.run([], input_dict)
                
                
                output_list = output_list
                caches['en_conv_cache'] = output_list[1]
                caches['de_conv_cache'] = output_list[2]
                caches['en_tra_cache'] = output_list[3]
                caches['de_tra_cache'] = output_list[4]
                caches['inter_cache_0'] = output_list[5]
                caches['inter_cache_1'] = output_list[6]
            
            frame_idx = warmup_end
            while frame_idx < num_frames and sample_count < num_samples:
                frame = spec[:, :, frame_idx:frame_idx+1, :]
                
                sample_id = f"sample_{sample_count:05d}"
                
                mix_path = input_dirs['mix'] / f"{sample_id}.npy"
                np.save(str(mix_path), frame)
                
                for cache_name, cache_value in caches.items():
                    cache_path = input_dirs[cache_name] / f"{sample_id}.npy"
                    np.save(str(cache_path), cache_value)
                
                input_dict = {'mix': frame}
                input_dict.update(caches)
                output_list = session.run([], input_dict)
                
                caches['en_conv_cache'] = output_list[1]
                caches['de_conv_cache'] = output_list[2]
                caches['en_tra_cache'] = output_list[3]
                caches['de_tra_cache'] = output_list[4]
                caches['inter_cache_0'] = output_list[5]
                caches['inter_cache_1'] = output_list[6]
                
                sample_count += 1
                pbar.update(1)

                frame_idx += skip_frames
            
        except Exception as e:
            print(f"\n警告: 处理 {audio_file.name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    pbar.close()
    print(f"\n>>> 成功生成 {sample_count} 个样本")
    return sample_count


def pack_to_zip(temp_dir, zip_dir, input_names):
    print("\n>>> 开始打包数据...")
    
    total_size = 0
    
    for input_name in tqdm(input_names, desc="打包输入"):
        input_dir = temp_dir / input_name
        zip_path = zip_dir / f"{input_name}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
            npy_files = sorted(input_dir.glob("*.npy"))
            
            for npy_file in npy_files:
                zipf.write(npy_file, npy_file.name)
        
        file_size = zip_path.stat().st_size
        total_size += file_size
        print(f"  ✓ {input_name:20s}: {len(npy_files):4d} 个样本 -> {file_size / 1024 / 1024:.2f} MB")
    
    print(f"\n>>> 总大小: {total_size / 1024 / 1024:.2f} MB")
    print(f">>> 所有数据已打包到: {zip_dir}")


def main():
    parser = argparse.ArgumentParser(description='生成 ONNX 模型量化校准数据 (taishan)')
    parser.add_argument('--onnx_model', type=str, 
                        default='onnx_models/gtcrn_optimized.onnx',
                        help='ONNX 模型路径')
    parser.add_argument('--audio_dir', type=str, default='test_wavs',
                        help='音频文件目录')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='生成的样本数量（默认: 100）')
    parser.add_argument('--output_dir', type=str, default='calibration_data',
                        help='输出目录（默认: calibration_data')
    parser.add_argument('--max_audio_files', type=int, default=None,
                        help='最大音频文件数量（默认: 无限制）')
    parser.add_argument('--skip_frames', type=int, default=10,
                        help='采样间隔（每隔多少帧采样一次，默认: 10）')
    parser.add_argument('--warmup_frames', type=int, default=20,
                        help='每个音频开始的 warmup 帧数（默认: 20）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.onnx_model):
        print(f"错误: ONNX 模型不存在: {args.onnx_model}")
        print("请先运行 export_onnx_taishan.py 导出模型")
        return
    
    input_names = [
        'mix',
        'en_conv_cache', 'de_conv_cache',
        'en_tra_cache', 'de_tra_cache',
        'inter_cache_0', 'inter_cache_1'
    ]
    
    temp_dir, zip_dir = prepare_output_dirs(args.output_dir)
    
    audio_files = collect_audio_files(args.audio_dir, args.max_audio_files)
    
    if len(audio_files) == 0:
        print("错误: 未找到音频文件！")
        return
    
    num_samples = generate_calibration_data_with_onnx(
        args.onnx_model, 
        audio_files, 
        args.num_samples, 
        temp_dir,
        skip_frames=args.skip_frames,
        warmup_frames=args.warmup_frames
    )
    
    pack_to_zip(temp_dir, zip_dir, input_names)
    
    print("\n>>> 清理临时文件...")
    shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 70)
    print("完成！")
    print(f"生成了 {len(input_names)} 个压缩包，每个包含 {num_samples} 个样本")
    print(f"数据位置: {zip_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
