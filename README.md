# gtcrn模型导出与AXERA转换说明
本项目用于将gtcrn语音降噪模型导出为ONNX，并支持转换axmodel格式。

## 依赖环境
已验证环境：python3.10，创建虚拟环境后安装依赖库
```bash
依赖库
pip install -r requirements.txt
```

## 导出ONNX模型
1. 确保模型权重文件已放置在 `onnx_models/` 目录下（如 `model_trained_on_dns3.tar`）。
2. 运行导出脚本：
```bash
sh export.sh
```
导出的ONNX文件 `onnx_models/gtcrn_optimized.onnx` 。

## 量化数据生成
```
python stream/generate_quantization_data_advanced.py --onnx_model onnx_models/gtcrn_optimized.onnx --audio_dir test_wavs --num_samples 100 --skip_frames 5 --warmup_frames 20
```
生成量化数据保存在`calibration_data`文件夹

## 模型转换（onnx->axmodel）
将导出的ONNX模型转换为AXERA平台的axmodel格式：
```bash
pulsar2 build --config config/config_gtcrn_615.json
```
量化模型保存在`output_620L`文件夹
