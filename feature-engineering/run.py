"""本地运行特征提取的命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extractor import build_features, save_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="US-05 木马样本特征提取")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
        help="配置文件路径，默认为本目录下的 config.json",
    )
    parser.add_argument("--bytes-dir", type=Path, help="覆盖 .bytes 文件目录")
    parser.add_argument("--labels", type=Path, help="覆盖标签 CSV 路径")
    parser.add_argument("--manifest", type=Path, help="覆盖预处理清单 CSV 路径")
    parser.add_argument("--output-dir", type=Path, help="覆盖输出目录")
    parser.add_argument("--max-samples", type=int, default=None, help="只处理前 N 个样本（用于快速验证）")
    parser.add_argument("--ngram-ns", default=None, help="逗号分隔的 N-gram 阶数，如 1,2,3")
    parser.add_argument("--top-k", default=None, help="逗号分隔的 top-k，与 --ngram-ns 顺序对应")
    parser.add_argument("--max-length", type=int, default=None, help="字节序列固定长度")
    parser.add_argument("--no-write", action="store_true", help="只计算不落盘")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)

    if args.bytes_dir is not None:
        config["input"]["bytes_dir"] = str(args.bytes_dir)
    if args.labels is not None:
        config["input"]["labels_csv"] = str(args.labels)
    if args.manifest is not None:
        config["input"]["manifest_csv"] = str(args.manifest)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples
    if args.max_length is not None:
        config["byte_sequence"]["max_length"] = args.max_length
    if args.ngram_ns is not None:
        ngram_ns = [int(item.strip()) for item in args.ngram_ns.split(",") if item.strip()]
        config["ngram"]["ns"] = ngram_ns
    if args.top_k is not None:
        top_k_values = [int(item.strip()) for item in args.top_k.split(",") if item.strip()]
        ngram_ns = config["ngram"]["ns"]
        if len(top_k_values) == 1:
            config["ngram"]["top_k"] = top_k_values[0]
        else:
            config["ngram"]["top_k"] = {
                str(n): value for n, value in zip(ngram_ns, top_k_values)
            }

    batch = build_features(config, base_dir=config_path.parent)
    if args.no_write:
        print(f"Computed {len(batch.ids)} samples in {batch.elapsed_seconds:.2f}s (not written).")
        return

    output_dir = batch.config["output_dir"]
    summary = save_outputs(batch, output_dir)
    print(f"Saved {summary['sample_count']} samples to {output_dir}.")
    print(f"Byte sequence shape: {summary['byte_sequence']['shape']}")
    print(f"Entropy shape: {summary['entropy']['shape']}")
    for n, info in summary["ngram"].items():
        print(f"N-gram {n} shape: {info['shape']}")
    print(f"Elapsed: {batch.elapsed_seconds:.2f}s")


if __name__ == "__main__":
    main()
