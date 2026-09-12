"""载入最佳Swin模型，识别清洗PNG或.bytes样本的家族，不执行样本。"""
import argparse
import json
import time
import torch
from data import FAMILIES, PREPROCESS, image_tensor, bytes_tensor
from models import SwinClassifier
from train import choose_device, sync


def predict(checkpoint_path, input_path, kind, device_name='auto'):
    device = choose_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint['families'] != FAMILIES or checkpoint['preprocess'] != PREPROCESS:
        raise ValueError('模型类别或预处理协议不一致')
    model = SwinClassifier(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    sync(device); start = time.perf_counter()
    x = bytes_tensor(input_path) if kind == 'bytes' else image_tensor(input_path)
    x = x.unsqueeze(0).to(device)
    sync(device); prepared = time.perf_counter()
    with torch.inference_mode():
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise ValueError('模型输出非有限值')
        scores = logits.softmax(1)[0].cpu()
    sync(device); end = time.perf_counter()
    values, indices = scores.topk(5)
    top5 = [{'class_id': int(i)+1, 'family': FAMILIES[int(i)], 'score': float(v)} for v,i in zip(values, indices)]
    result = {'predicted_family': top5[0]['family'], 'top5': top5,
              'preprocess_seconds': prepared-start, 'cold_prediction_seconds': end-prepared,
              'notice': '得分未经校准，不代表恶意概率；仅识别9类家族。PNG须使用同一清洗规则生成。计时不含模型加载。'}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--kind', choices=['png', 'bytes'], required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    predict(args.checkpoint, args.input, args.kind, args.device)
