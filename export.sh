#!/bin/bash

echo "Step 1: 正在导出ONNX模型..."
python stream/export_onnx.py 

echo "Step 2: 正在优化ONNX模型..."
python pass_clear_gather.py onnx_models/gtcrn_simple.onnx onnx_models/gtcrn_optimized.onnx

echo "导出完成！"