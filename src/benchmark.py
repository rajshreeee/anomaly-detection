import argparse
import logging
import time

import faiss
import numpy as np
import pandas as pd
import torch

import config
from efficient_ad.inference import _load_models, _get_anomaly_map
from patchcore.inference   import _extract_features, _score_map
from pathlib import Path

logger = logging.getLogger(__name__)


class BenchmarkTimer:
    def __init__(self, device: str):
        self.device = device
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start_time

def parse_args():
    parser = argparse.ArgumentParser(description="EfficientAD vs PatchCore inference benchmark")
    parser.add_argument("--reps",       type=int, default=100,
                        help="Timed repetitions after warmup (default: 100)")
    parser.add_argument("--warmup",     type=int, default=10,
                        help="Warmup reps, results discarded (default: 10)")
    parser.add_argument("--index_path", default=config.PC_MEMORY_BANK_FAISS,
                        help="PatchCore .faiss index path (default: config.PC_MEMORY_BANK_FAISS)")
    return parser.parse_args()

def _bench_ead(reps: int, warmup: int, device: str) -> np.ndarray:
    (teacher, student, autoencoder,
     t_mean, t_std,
     q_st_start, q_st_end,
     q_ae_start, q_ae_end) = _load_models(device)


    times = []
    with torch.no_grad():
        for i in range(reps + warmup):
            img = torch.randn(1, 3, config.EAD_IMAGE_SIZE, config.EAD_IMAGE_SIZE)
            with BenchmarkTimer(device) as t:
                _ = _get_anomaly_map(
                    img, teacher, student, autoencoder,
                    t_mean, t_std,
                    q_st_start, q_st_end,
                    q_ae_start, q_ae_end,
                    device,
                )
            if i >= warmup:
                times.append(t.elapsed)

    return np.array(times)


def _bench_pc(reps: int, warmup: int, device: str, index_path: str) -> np.ndarray:
    from patchcore.model import FeatureExtractor
    extractor = FeatureExtractor(layers=config.PC_LAYERS, device=device)

    index = faiss.read_index(index_path)
    if device == "cuda" and faiss.get_num_gpus() > 0:
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)


    times = []
    with torch.no_grad():
        for i in range(reps + warmup):
            img = torch.randn(1, 3, config.PC_IMAGE_SIZE, config.PC_IMAGE_SIZE)
            with BenchmarkTimer(device) as t:
                features, ref_shape = _extract_features(img, extractor, device)
                dists, _            = index.search(features, 1)
                _                   = _score_map(dists, ref_shape, config.PC_IMAGE_SIZE)

            if i >= warmup:
                times.append(t.elapsed)
    return np.array(times)


def _save_results(all_results: dict, out_dir: Path) -> None:
    rows = []
    for name, times in all_results.items():
        ms = times * 1000
        rows.append({
            "Model"     : name,
            "Mean (ms)" : round(ms.mean(),       2),
            "Min (ms)"  : round(ms.min(),        2),
            "Max (ms)"  : round(ms.max(),        2),
        })
    df = pd.DataFrame(rows).set_index("Model")
    df.to_csv(out_dir / "benchmark_results.csv")
    logger.info("Saved benchmark_results.csv")

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
    args = parse_args()
    
    out_dir = Path(config.RESULTS_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for device in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
        logger.info(f"── {device.upper()} ──")
        all_results[f"EfficientAD ({device})"] = _bench_ead(args.reps, args.warmup, device)
        all_results[f"PatchCore ({device})"]   = _bench_pc(args.reps, args.warmup, device, args.index_path)
    _save_results(all_results, out_dir)

if __name__ == "__main__":
    main()
