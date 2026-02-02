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
    # 读取音频
    audio, sr = sf.read(str(audio_path), dtype='float32')
    
    # 如果是立体声，转换为单声道
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # 重采样到 16kHz
    if sr != 16000:
        print(f"\n警告: {audio_path.name} 采样率为 {sr}，期望 16000")
    
    # STFT
    audio_tensor = torch.from_numpy(audio)
    spec = torch.stft(
        audio_tensor, 
        n_fft=512, 
        hop_length=256, 
        win_length=512, 
        window=torch.hann_window(512).pow(0.5), 
        return_complex=False
    )
    
    # 添加 batch 维度: (F, T, 2) -> (1, F, T, 2)
    spec = spec.unsqueeze(0).numpy()
    
    return spec


def init_caches():
    return {
        'en_conv_cache_0': np.zeros([1, 16, 2, 33], dtype="float32"),
        'en_conv_cache_1': np.zeros([1, 16, 4, 33], dtype="float32"),
        'en_conv_cache_2': np.zeros([1, 16, 10, 33], dtype="float32"),
        'de_conv_cache_0': np.zeros([1, 16, 10, 33], dtype="float32"),
        'de_conv_cache_1': np.zeros([1, 16, 4, 33], dtype="float32"),
        'de_conv_cache_2': np.zeros([1, 16, 2, 33], dtype="float32"),
        'en_tra_cache_0': np.zeros([1, 1, 16], dtype="float32"),
        'en_tra_cache_1': np.zeros([1, 1, 16], dtype="float32"),
        'en_tra_cache_2': np.zeros([1, 1, 16], dtype="float32"),
        'de_tra_cache_0': np.zeros([1, 1, 16], dtype="float32"),
        'de_tra_cache_1': np.zeros([1, 1, 16], dtype="float32"),
        'de_tra_cache_2': np.zeros([1, 1, 16], dtype="float32"),
        'inter_cache_0': np.zeros([1, 33, 16], dtype="float32"),
        'inter_cache_1': np.zeros([1, 33, 16], dtype="float32"),
    }


def generate_calibration_data_with_onnx(onnx_model_path, audio_files, num_samples, 
                                         temp_dir, skip_frames=1, warmup_frames=10):
    
    input_names = [
        'mix',
        'en_conv_cache_0', 'en_conv_cache_1', 'en_conv_cache_2',
        'de_conv_cache_0', 'de_conv_cache_1', 'de_conv_cache_2',
        'en_tra_cache_0', 'en_tra_cache_1', 'en_tra_cache_2',
        'de_tra_cache_0', 'de_tra_cache_1', 'de_tra_cache_2',
        'inter_cache_0', 'inter_cache_1'
    ]
    
    input_dirs = {}
    for name in input_names:
        input_dir = temp_dir / name
        input_dir.mkdir(exist_ok=True)
        input_dirs[name] = input_dir
    
    # 加载 ONNX 模型
    print(f"\n>>> 加载 ONNX 模型: {onnx_model_path}")
    session = onnxruntime.InferenceSession(
        onnx_model_path, 
        None,
        providers=['CPUExecutionProvider']
    )
    
    sample_count = 0
    
    print(f"\n>>> 开始生成校准数据，目标样本数: {num_samples}")
    print(f">>> 跳帧间隔: {skip_frames} (warmup: {warmup_frames} 帧)")
    
    # 遍历音频文件
    pbar = tqdm(total=num_samples, desc="生成样本")
    
    for audio_idx, audio_file in enumerate(audio_files):
        if sample_count >= num_samples:
            break
        
        try:
            # 转换为频谱
            spec = audio_to_spec(audio_file)
            num_frames = spec.shape[2]
            
            # 初始化 cache
            caches = init_caches()
            
            # Warmup 阶段：让 cache 达到稳定状态
            warmup_end = min(warmup_frames, num_frames)
            for frame_idx in range(warmup_end):
                frame = spec[:, :, frame_idx:frame_idx+1, :]
                
                # 准备输入
                input_dict = {'mix': frame}
                input_dict.update(caches)
                
                # 推理
                output_list = session.run([], input_dict)
                
                # 更新 cache（output_list[0] 是 enh，后面是 caches）
                cache_names = list(caches.keys())
                for i, cache_name in enumerate(cache_names):
                    caches[cache_name] = output_list[i + 1]
            
            # 采样阶段：每隔 skip_frames 保存一次
            frame_idx = warmup_end
            while frame_idx < num_frames and sample_count < num_samples:
                frame = spec[:, :, frame_idx:frame_idx+1, :]
                
                # 保存当前帧的输入数据（推理前的状态）
                sample_id = f"sample_{sample_count:05d}"
                
                # 保存 mix
                mix_path = input_dirs['mix'] / f"{sample_id}.npy"
                np.save(str(mix_path), frame)
                
                # 保存当前的 cache 状态
                for cache_name, cache_value in caches.items():
                    cache_path = input_dirs[cache_name] / f"{sample_id}.npy"
                    np.save(str(cache_path), cache_value)
                
                # 推理并更新 cache
                input_dict = {'mix': frame}
                input_dict.update(caches)
                output_list = session.run([], input_dict)
                
                # 更新 cache
                cache_names = list(caches.keys())
                for i, cache_name in enumerate(cache_names):
                    caches[cache_name] = output_list[i + 1]
                
                sample_count += 1
                pbar.update(1)
                
                # 跳到下一个采样点
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
        
        # 创建压缩包
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
            npy_files = sorted(input_dir.glob("*.npy"))
            
            for npy_file in npy_files:
                # 只保存文件名，不保存路径
                zipf.write(npy_file, npy_file.name)
        
        file_size = zip_path.stat().st_size
        total_size += file_size
        print(f"  ✓ {input_name:20s}: {len(npy_files):4d} 个样本 -> {file_size / 1024 / 1024:.2f} MB")
    
    print(f"\n>>> 总大小: {total_size / 1024 / 1024:.2f} MB")
    print(f">>> 所有数据已打包到: {zip_dir}")


def generate_data_info(zip_dir, num_samples, input_names, args):
    info_path = zip_dir / "README.txt"
    
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ONNX 模型量化校准数据\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("生成参数:\n")
        f.write(f"  - ONNX 模型: {args.onnx_model}\n")
        f.write(f"  - 样本数量: {num_samples}\n")
        f.write(f"  - 跳帧间隔: {args.skip_frames}\n")
        f.write(f"  - Warmup 帧数: {args.warmup_frames}\n")
        f.write(f"  - 音频目录: {args.audio_dir}\n\n")
        
        f.write("数据结构:\n")
        f.write(f"  - 输入数量: {len(input_names)}\n")
        f.write(f"  - 每个输入: {num_samples} 个样本 (.npy 格式)\n\n")
        
        f.write("压缩包列表:\n")
        for i, name in enumerate(input_names, 1):
            zip_file = zip_dir / f"{name}.zip"
            if zip_file.exists():
                size_mb = zip_file.stat().st_size / (1024 * 1024)
                f.write(f"  {i:2d}. {name:20s} -> {name}.zip ({size_mb:.2f} MB)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("使用方法:\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. 解压压缩包:\n")
        f.write("   unzip mix.zip -d mix/\n")
        f.write("   unzip en_conv_cache_0.zip -d en_conv_cache_0/\n")
        f.write("   ...\n\n")
        
        f.write("2. 读取样本数据:\n")
        f.write("   import numpy as np\n")
        f.write("   mix_sample_0 = np.load('mix/sample_00000.npy')\n")
        f.write("   en_cache_0 = np.load('en_conv_cache_0/sample_00000.npy')\n\n")
        
        f.write("3. 使用 Python 脚本批量读取:\n")
        f.write("   见 load_calibration_data.py\n\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"\n>>> 数据信息已保存到: {info_path}")


def generate_loader_script(zip_dir, input_names, num_samples):
    loader_path = zip_dir / "load_calibration_data.py"
    
    with open(loader_path, 'w') as f:
        f.write('"""\n')
        f.write('校准数据加载脚本\n\n')
        f.write('使用方法:\n')
        f.write('    from load_calibration_data import load_sample\n')
        f.write('    data = load_sample(0)  # 加载第 0 个样本\n')
        f.write('"""\n\n')
        f.write('import numpy as np\n')
        f.write('from pathlib import Path\n')
        f.write('import zipfile\n\n\n')
        
        f.write('INPUT_NAMES = [\n')
        for name in input_names:
            f.write(f"    '{name}',\n")
        f.write(']\n\n\n')
        
        f.write('def extract_all(data_dir="."):\n')
        f.write('    """解压所有压缩包"""\n')
        f.write('    data_dir = Path(data_dir)\n')
        f.write('    for name in INPUT_NAMES:\n')
        f.write('        zip_path = data_dir / f"{name}.zip"\n')
        f.write('        extract_dir = data_dir / name\n')
        f.write('        if zip_path.exists() and not extract_dir.exists():\n')
        f.write('            print(f"解压 {name}.zip...")\n')
        f.write('            with zipfile.ZipFile(zip_path, "r") as zipf:\n')
        f.write('                zipf.extractall(extract_dir)\n\n\n')
        
        f.write('def load_sample(sample_idx, data_dir="."):\n')
        f.write('    """加载单个样本的所有输入\n    \n')
        f.write('    Args:\n')
        f.write('        sample_idx: 样本索引 (0 到 {})\n'.format(num_samples - 1))
        f.write('        data_dir: 数据目录路径\n    \n')
        f.write('    Returns:\n')
        f.write('        dict: {input_name: numpy_array}\n')
        f.write('    """\n')
        f.write('    data_dir = Path(data_dir)\n')
        f.write('    sample_data = {}\n')
        f.write('    \n')
        f.write('    for name in INPUT_NAMES:\n')
        f.write('        npy_path = data_dir / name / f"sample_{sample_idx:05d}.npy"\n')
        f.write('        if npy_path.exists():\n')
        f.write('            sample_data[name] = np.load(str(npy_path))\n')
        f.write('        else:\n')
        f.write('            raise FileNotFoundError(f"未找到: {npy_path}")\n')
        f.write('    \n')
        f.write('    return sample_data\n\n\n')
        
        f.write('def load_all_samples(data_dir="."):\n')
        f.write('    """加载所有样本\n    \n')
        f.write('    Returns:\n')
        f.write('        list of dict: 每个样本的数据字典列表\n')
        f.write('    """\n')
        f.write(f'    num_samples = {num_samples}\n')
        f.write('    samples = []\n')
        f.write('    \n')
        f.write('    for i in range(num_samples):\n')
        f.write('        samples.append(load_sample(i, data_dir))\n')
        f.write('    \n')
        f.write('    return samples\n\n\n')
        
        f.write('if __name__ == "__main__":\n')
        f.write('    # 示例用法\n')
        f.write('    print("解压所有数据...")\n')
        f.write('    extract_all()\n')
        f.write('    \n')
        f.write('    print("\\n加载第一个样本...")\n')
        f.write('    sample_0 = load_sample(0)\n')
        f.write('    \n')
        f.write('    print("\\n数据形状:")\n')
        f.write('    for name, data in sample_0.items():\n')
        f.write('        print(f"  {name:20s}: {data.shape}")\n')
    
    print(f">>> 加载脚本已保存到: {loader_path}")


def main():
    parser = argparse.ArgumentParser(description='生成 ONNX 模型量化校准数据')
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
        print("请先运行 export_onnx_no_scatter.py 导出模型")
        return
    
    # print("=" * 70)
    # print("ONNX 模型量化校准数据生成器")
    # print("=" * 70)
    # print(f"ONNX 模型: {args.onnx_model}")
    # print(f"音频目录: {args.audio_dir}")
    # print(f"目标样本数: {args.num_samples}")
    # print(f"跳帧间隔: {args.skip_frames}")
    # print(f"Warmup 帧数: {args.warmup_frames}")
    # print(f"输出目录: {args.output_dir}")
    # print("=" * 70)
    
    input_names = [
        'mix',
        'en_conv_cache_0', 'en_conv_cache_1', 'en_conv_cache_2',
        'de_conv_cache_0', 'de_conv_cache_1', 'de_conv_cache_2',
        'en_tra_cache_0', 'en_tra_cache_1', 'en_tra_cache_2',
        'de_tra_cache_0', 'de_tra_cache_1', 'de_tra_cache_2',
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
    
    generate_data_info(zip_dir, num_samples, input_names, args)
    
    generate_loader_script(zip_dir, input_names, num_samples)
    
    print("\n>>> 清理临时文件...")
    shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 70)
    print("完成！")
    print(f"生成了 {len(input_names)} 个压缩包，每个包含 {num_samples} 个样本")
    print(f"数据位置: {zip_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
