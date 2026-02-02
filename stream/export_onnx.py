import torch
import numpy as np
import torch.nn as nn
from stream.gtcrn_stream import ERB, SFE, StreamTRA, ConvBlock, StreamGTConvBlock, GRNN, DPGRNN, Mask


class StreamEncoderNoScatter(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, 16, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
        ])

    def forward(self, x, conv_cache_0, conv_cache_1, conv_cache_2, tra_cache_0, tra_cache_1, tra_cache_2):
        en_outs = []
        for i in range(2):
            x = self.en_convs[i](x)
            en_outs.append(x)
        
        # 分别处理每个 GTConvBlock，返回独立的 cache
        x, conv_cache_0, tra_cache_0 = self.en_convs[2](x, conv_cache_0, tra_cache_0)
        en_outs.append(x)
        
        x, conv_cache_1, tra_cache_1 = self.en_convs[3](x, conv_cache_1, tra_cache_1)
        en_outs.append(x)
        
        x, conv_cache_2, tra_cache_2 = self.en_convs[4](x, conv_cache_2, tra_cache_2)
        en_outs.append(x)
            
        return x, en_outs, conv_cache_0, conv_cache_1, conv_cache_2, tra_cache_0, tra_cache_1, tra_cache_2


class StreamDecoderNoScatter(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=True),
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=True),
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=True),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(16, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs, conv_cache_0, conv_cache_1, conv_cache_2, tra_cache_0, tra_cache_1, tra_cache_2):
        x, conv_cache_0, tra_cache_0 = self.de_convs[0](x + en_outs[4], conv_cache_0, tra_cache_0)
        x, conv_cache_1, tra_cache_1 = self.de_convs[1](x + en_outs[3], conv_cache_1, tra_cache_1)
        x, conv_cache_2, tra_cache_2 = self.de_convs[2](x + en_outs[2], conv_cache_2, tra_cache_2)
        
        for i in range(3, 5):
            x = self.de_convs[i](x + en_outs[4-i])
            
        return x, conv_cache_0, conv_cache_1, conv_cache_2, tra_cache_0, tra_cache_1, tra_cache_2


class StreamGTCRNNoScatter(nn.Module):
    def __init__(self):
        super().__init__()
        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)
        self.encoder = StreamEncoderNoScatter()
        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)
        self.decoder = StreamDecoderNoScatter()
        self.mask = Mask()

    def forward(self, spec, 
                en_conv_cache_0, en_conv_cache_1, en_conv_cache_2,
                de_conv_cache_0, de_conv_cache_1, de_conv_cache_2,
                en_tra_cache_0, en_tra_cache_1, en_tra_cache_2,
                de_tra_cache_0, de_tra_cache_1, de_tra_cache_2,
                inter_cache_0, inter_cache_1):

        spec_ref = spec  # (B,F,T,2)

        spec_real = spec[..., 0].permute(0,2,1)
        spec_imag = spec[..., 1].permute(0,2,1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)

        feat = self.erb.bm(feat)  # (B,3,T,129)
        feat = self.sfe(feat)     # (B,9,T,129)

        feat, en_outs, en_conv_cache_0, en_conv_cache_1, en_conv_cache_2, \
            en_tra_cache_0, en_tra_cache_1, en_tra_cache_2 = \
            self.encoder(feat, en_conv_cache_0, en_conv_cache_1, en_conv_cache_2,
                        en_tra_cache_0, en_tra_cache_1, en_tra_cache_2)

        # DPGRNN
        feat, inter_cache_0 = self.dpgrnn1(feat, inter_cache_0)
        feat, inter_cache_1 = self.dpgrnn2(feat, inter_cache_1)

        m_feat, de_conv_cache_0, de_conv_cache_1, de_conv_cache_2, \
            de_tra_cache_0, de_tra_cache_1, de_tra_cache_2 = \
            self.decoder(feat, en_outs, de_conv_cache_0, de_conv_cache_1, de_conv_cache_2,
                        de_tra_cache_0, de_tra_cache_1, de_tra_cache_2)
        
        m = self.erb.bs(m_feat)
        spec_enh = self.mask(m, spec_ref.permute(0,3,2,1))
        spec_enh = spec_enh.permute(0,3,2,1)
        return (spec_enh, 
                en_conv_cache_0, en_conv_cache_1, en_conv_cache_2,
                de_conv_cache_0, de_conv_cache_1, de_conv_cache_2,
                en_tra_cache_0, en_tra_cache_1, en_tra_cache_2,
                de_tra_cache_0, de_tra_cache_1, de_tra_cache_2,
                inter_cache_0, inter_cache_1)


def convert_to_no_scatter_model(no_scatter_model, original_model):

    no_scatter_state = no_scatter_model.state_dict()
    original_state = original_model.state_dict()
    
    for key in no_scatter_state.keys():
        if key in original_state.keys():
            no_scatter_state[key] = original_state[key]
    
    no_scatter_model.load_state_dict(no_scatter_state)
    print(">>> 模型权重转换完成")


if __name__ == "__main__":
    import os
    import time
    import soundfile as sf
    from tqdm import tqdm
    import onnx
    import onnxruntime
    from onnxsim import simplify
    from librosa import istft
    from stream.gtcrn import GTCRN
    from stream.modules.convert import convert_to_stream
    from stream.gtcrn_stream import StreamGTCRN
    
    device = torch.device("cpu")
    
    model = GTCRN().to(device).eval()
    model.load_state_dict(torch.load('onnx_models/model_trained_on_dns3.tar', 
                                     map_location=device)['model'])
    
    stream_model_original = StreamGTCRN().to(device).eval()
    convert_to_stream(stream_model_original, model)
    
    stream_model_no_scatter = StreamGTCRNNoScatter().to(device).eval()
    convert_to_no_scatter_model(stream_model_no_scatter, stream_model_original)
    
    x = torch.from_numpy(sf.read('test_wavs/mix.wav', dtype='float32')[0])
    x = torch.stft(x, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)[None]
    
    with torch.no_grad():
        y = model(x)
    y_complex = torch.view_as_complex(y.contiguous())
    y = torch.istft(y_complex, 512, 256, 512, torch.hann_window(512).pow(0.5)).detach().cpu().numpy()
    
    en_conv_cache_0 = torch.zeros(1, 16, 2, 33).to(device)
    en_conv_cache_1 = torch.zeros(1, 16, 4, 33).to(device)
    en_conv_cache_2 = torch.zeros(1, 16, 10, 33).to(device)
    de_conv_cache_0 = torch.zeros(1, 16, 10, 33).to(device)
    de_conv_cache_1 = torch.zeros(1, 16, 4, 33).to(device)
    de_conv_cache_2 = torch.zeros(1, 16, 2, 33).to(device)
    
    # TRA cache size is (1, B, channels*2) where channels = in_channels//2 = 8, so channels*2 = 16
    en_tra_cache_0 = torch.zeros(1, 1, 16).to(device)
    en_tra_cache_1 = torch.zeros(1, 1, 16).to(device)
    en_tra_cache_2 = torch.zeros(1, 1, 16).to(device)
    de_tra_cache_0 = torch.zeros(1, 1, 16).to(device)
    de_tra_cache_1 = torch.zeros(1, 1, 16).to(device)
    de_tra_cache_2 = torch.zeros(1, 1, 16).to(device)
    
    inter_cache_0 = torch.zeros(1, 33, 16).to(device)
    inter_cache_1 = torch.zeros(1, 33, 16).to(device)
    
    ys = []
    for i in tqdm(range(x.shape[2]), desc="PyTorch 推理"):
        xi = x[:,:,i:i+1]
        with torch.no_grad():
            outputs = stream_model_no_scatter(
                xi, 
                en_conv_cache_0, en_conv_cache_1, en_conv_cache_2,
                de_conv_cache_0, de_conv_cache_1, de_conv_cache_2,
                en_tra_cache_0, en_tra_cache_1, en_tra_cache_2,
                de_tra_cache_0, de_tra_cache_1, de_tra_cache_2,
                inter_cache_0, inter_cache_1
            )
            yi = outputs[0]
            # 更新所有 cache
            (en_conv_cache_0, en_conv_cache_1, en_conv_cache_2,
             de_conv_cache_0, de_conv_cache_1, de_conv_cache_2,
             en_tra_cache_0, en_tra_cache_1, en_tra_cache_2,
             de_tra_cache_0, de_tra_cache_1, de_tra_cache_2,
             inter_cache_0, inter_cache_1) = outputs[1:]
        ys.append(yi)
    
    ys = torch.cat(ys, dim=2)

    ys_complex = torch.view_as_complex(ys.contiguous())
    ys = torch.istft(ys_complex, 512, 256, 512, torch.hann_window(512).pow(0.5)).detach().cpu().numpy()
    # sf.write('test_wavs/enh_pt.wav', ys.squeeze(), 16000)
    
    output_file = 'onnx_models/gtcrn.onnx'
    
    # 准备示例输入
    dummy_input = torch.randn(1, 257, 1, 2, device=device)
    dummy_caches = (
        en_conv_cache_0, en_conv_cache_1, en_conv_cache_2,
        de_conv_cache_0, de_conv_cache_1, de_conv_cache_2,
        en_tra_cache_0, en_tra_cache_1, en_tra_cache_2,
        de_tra_cache_0, de_tra_cache_1, de_tra_cache_2,
        inter_cache_0, inter_cache_1
    )
    
    # 导出 ONNX
    torch.onnx.export(
        stream_model_no_scatter,
        (dummy_input,) + dummy_caches,
        output_file,
        input_names=[
            'mix',
            'en_conv_cache_0', 'en_conv_cache_1', 'en_conv_cache_2',
            'de_conv_cache_0', 'de_conv_cache_1', 'de_conv_cache_2',
            'en_tra_cache_0', 'en_tra_cache_1', 'en_tra_cache_2',
            'de_tra_cache_0', 'de_tra_cache_1', 'de_tra_cache_2',
            'inter_cache_0', 'inter_cache_1'
        ],
        output_names=[
            'enh',
            'en_conv_cache_0_out', 'en_conv_cache_1_out', 'en_conv_cache_2_out',
            'de_conv_cache_0_out', 'de_conv_cache_1_out', 'de_conv_cache_2_out',
            'en_tra_cache_0_out', 'en_tra_cache_1_out', 'en_tra_cache_2_out',
            'de_tra_cache_0_out', 'de_tra_cache_1_out', 'de_tra_cache_2_out',
            'inter_cache_0_out', 'inter_cache_1_out'
        ],
        opset_version=11,
        verbose=False
    )
    
    print(f">>> ONNX 模型已导出到: {output_file}")

    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)
    
    simplified_file = output_file.replace('.onnx', '_simple.onnx')
    if not os.path.exists(simplified_file):
        model_simp, check = simplify(onnx_model)
        assert check, "简化后的 ONNX 模型验证失败"
        onnx.save(model_simp, simplified_file)
        print(f">>> 简化模型已保存到: {simplified_file}")

    # simplified_file = "onnx_models/gtcrn_optimized.onnx"
    
    session = onnxruntime.InferenceSession(simplified_file, None, 
                                          providers=['CPUExecutionProvider'])
    
    cache_dict = {
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
    
    T_list = []
    outputs = []
    inputs = x.numpy()
    
    for i in tqdm(range(inputs.shape[-2]), desc="ONNX Runtime 推理"):
        tic = time.perf_counter()
        
        input_dict = {'mix': inputs[..., i:i+1, :]}
        input_dict.update(cache_dict)
        
        output_list = session.run([], input_dict)
        
        toc = time.perf_counter()
        T_list.append(toc-tic)
        
        outputs.append(output_list[0])
        
        cache_dict = {
            'en_conv_cache_0': output_list[1],
            'en_conv_cache_1': output_list[2],
            'en_conv_cache_2': output_list[3],
            'de_conv_cache_0': output_list[4],
            'de_conv_cache_1': output_list[5],
            'de_conv_cache_2': output_list[6],
            'en_tra_cache_0': output_list[7],
            'en_tra_cache_1': output_list[8],
            'en_tra_cache_2': output_list[9],
            'de_tra_cache_0': output_list[10],
            'de_tra_cache_1': output_list[11],
            'de_tra_cache_2': output_list[12],
            'inter_cache_0': output_list[13],
            'inter_cache_1': output_list[14],
        }
    
    outputs = np.concatenate(outputs, axis=2)
    enhanced = istft(outputs[...,0] + 1j * outputs[...,1], 
                    n_fft=512, hop_length=256, win_length=512, 
                    window=np.hanning(512)**0.5)
    # sf.write('test_wavs/enh_onnx.wav', enhanced.squeeze(), 16000)
    
    print(f"\n{'='*60}")
    print(">>> 测试结果:")
    print(f">>> ONNX 推理误差: {np.abs(y - enhanced).max():.6f}")
    print(f">>> 推理时间 - 平均: {1e3*np.mean(T_list):.2f}ms, "
          f"最大: {1e3*np.max(T_list):.2f}ms, 最小: {1e3*np.min(T_list):.2f}ms")
    print(f">>> RTF (Real-Time Factor): {1e3*np.mean(T_list) / 16:.4f}")
    print(f"{'='*60}")
    print("\n>>> 检查 ONNX 模型算子:")
    node_types = set([node.op_type for node in onnx_model.graph.node])
    
    print(f"\n模型包含的算子类型: {sorted(node_types)}")
