import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def build_text_for_embedding(record: Dict) -> str:
    if "feature_content" in record and str(record["feature_content"]).strip():
        return str(record["feature_content"]).strip()

    if "instruction" in record or "input" in record or "output" in record:
        parts = []
        if str(record.get("instruction", "")).strip():
            parts.append(f"instruction: {record.get('instruction', '')}")
        if str(record.get("input", "")).strip():
            parts.append(f"input: {record.get('input', '')}")
        if str(record.get("output", "")).strip():
            parts.append(f"output: {record.get('output', '')}")
        if parts:
            return "\n".join(parts)

    if "question" in record:
        parts = [f"question: {record.get('question', '')}"]
        for key in ["A", "B", "C", "D", "E", "answer", "explanation"]:
            if key in record and str(record.get(key, "")).strip():
                parts.append(f"{key}: {record.get(key)}")
        return "\n".join(parts)

    parts = []
    for k, v in record.items():
        if isinstance(v, (str, int, float, bool)):
            text = str(v).strip()
            if text:
                parts.append(f"{k}: {text}")
    return "\n".join(parts)


def maybe_truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    return text[:max_chars]


def iter_nonempty_jsonl(path: str, start_from_nonempty_idx: int = 0) -> Iterable[Tuple[int, Dict]]:
    seen_nonempty = 0
    with open(path, "r", encoding="utf-8") as f:
        for raw_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if seen_nonempty < start_from_nonempty_idx:
                seen_nonempty += 1
                continue
            yield raw_idx, json.loads(line)
            seen_nonempty += 1


def count_nonempty_lines(path: str) -> int:
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def batch_iter(iterable: Iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def save_checkpoint(path: str, state: Dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_device(user_device: Optional[str]) -> str:
    if user_device:
        return user_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_or_build_val_matrix(
    model: SentenceTransformer,
    val_path: str,
    encode_batch_size: int,
    text_max_chars: int,
    save_val_vectors_path: str,
    reuse_val_vectors: bool,
) -> np.ndarray:
    if reuse_val_vectors and os.path.exists(save_val_vectors_path):
        vecs = []
        with open(save_val_vectors_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                vec = row.get("feature_vector")
                if vec is None:
                    vecs = []
                    break
                vecs.append(vec)
        if vecs:
            print(f"Loaded val vectors from {save_val_vectors_path}")
            return np.array(vecs, dtype=np.float32)

    val_rows = []
    val_texts = []
    for _, row in iter_nonempty_jsonl(val_path):
        val_rows.append(row)
        val_texts.append(maybe_truncate(build_text_for_embedding(row), text_max_chars))

    val_emb = encode_texts(model, val_texts, batch_size=encode_batch_size)
    with open(save_val_vectors_path, "w", encoding="utf-8") as fout:
        for i, row in enumerate(val_rows):
            row_with_vec = dict(row)
            row_with_vec["feature_vector"] = val_emb[i].tolist()
            fout.write(json.dumps(row_with_vec, ensure_ascii=False) + "\n")
    print(f"Saved val vectors to {save_val_vectors_path}")
    return val_emb


def compute_avg_scores(train_emb: np.ndarray, val_emb: np.ndarray, topk: int) -> np.ndarray:
    sim = np.dot(train_emb, val_emb.T)
    idx = np.argpartition(-sim, topk - 1, axis=1)[:, :topk]
    top_scores = np.take_along_axis(sim, idx, axis=1)
    return top_scores.mean(axis=1)


def compute_and_save_avg_scores(
    model: SentenceTransformer,
    train_path: str,
    val_emb: np.ndarray,
    encode_batch_size: int,
    text_max_chars: int,
    topk: int,
    avg_scores_output_path: str,
    checkpoint_path: str,
    checkpoint_every_batches: int,
    resume: bool,
):
    if resume:
        ckpt = load_checkpoint(checkpoint_path)
        existing_lines = count_nonempty_lines(avg_scores_output_path) if os.path.exists(avg_scores_output_path) else 0
        if ckpt:
            processed_nonempty = int(ckpt.get("processed_nonempty_lines", 0))
            batch_index = int(ckpt.get("batch_index", 0))
            # Prefer the larger progress to avoid duplicate writes when checkpoint is stale.
            if existing_lines > processed_nonempty:
                processed_nonempty = existing_lines
        else:
            if existing_lines == 0:
                raise RuntimeError(
                    f"Resume requested but neither checkpoint nor score file exists: {checkpoint_path}, {avg_scores_output_path}"
                )
            processed_nonempty = existing_lines
            batch_index = 0
        out_mode = "a"
    else:
        processed_nonempty = 0
        batch_index = 0
        out_mode = "w"
        for p in [avg_scores_output_path, checkpoint_path]:
            if os.path.exists(p):
                os.remove(p)

    total = count_nonempty_lines(train_path)
    progress = tqdm(total=total, initial=min(processed_nonempty, total), desc="Computing avg_score")
    nonempty_counter = processed_nonempty

    with open(avg_scores_output_path, out_mode, encoding="utf-8") as fout:
        stream = iter_nonempty_jsonl(train_path, start_from_nonempty_idx=processed_nonempty)
        for batch in batch_iter(stream, encode_batch_size):
            row_ids = []
            texts = []
            for raw_idx, row in batch:
                row_id = row.get("id", raw_idx)
                row_ids.append(int(row_id))
                texts.append(maybe_truncate(build_text_for_embedding(row), text_max_chars))

            emb = encode_texts(model, texts, batch_size=encode_batch_size)
            avg_scores = compute_avg_scores(emb, val_emb, topk=topk)

            for i, row_id in enumerate(row_ids):
                fout.write(
                    json.dumps(
                        {"id": row_id, "avg_score": float(avg_scores[i])},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                nonempty_counter += 1

            batch_index += 1
            progress.update(len(row_ids))

            if checkpoint_every_batches > 0 and batch_index % checkpoint_every_batches == 0:
                fout.flush()
                save_checkpoint(
                    checkpoint_path,
                    {
                        "processed_nonempty_lines": nonempty_counter,
                        "batch_index": batch_index,
                        "avg_scores_output_path": avg_scores_output_path,
                        "status": "running",
                    },
                )

    progress.close()
    save_checkpoint(
        checkpoint_path,
        {
            "processed_nonempty_lines": nonempty_counter,
            "batch_index": batch_index,
            "avg_scores_output_path": avg_scores_output_path,
            "status": "completed",
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute avg_score only and save lightweight score file (id, avg_score)."
    )
    parser.add_argument("--train_path", default="train.jsonl")
    parser.add_argument("--val_path", default="val.jsonl")
    parser.add_argument("--model_name_or_path", default="BAAI/bge-m3")
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--encode_batch_size", type=int, default=48)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--text_max_chars", type=int, default=6000)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--save_val_vectors_path", default="val_with_bge_m3_vectors.jsonl")
    parser.add_argument("--reuse_val_vectors", action="store_true")
    parser.add_argument("--avg_scores_output_path", default="train_avg_scores_bge_m3.jsonl")
    parser.add_argument("--checkpoint_path", default="train_avg_scores_checkpoint.json")
    parser.add_argument("--checkpoint_every_batches", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model = SentenceTransformer(args.model_name_or_path, device=device)
    if args.max_seq_length > 0:
        model.max_seq_length = args.max_seq_length
    if args.fp16 and device.startswith("cuda"):
        model.half()

    val_emb = load_or_build_val_matrix(
        model=model,
        val_path=args.val_path,
        encode_batch_size=args.encode_batch_size,
        text_max_chars=args.text_max_chars,
        save_val_vectors_path=args.save_val_vectors_path,
        reuse_val_vectors=args.reuse_val_vectors,
    )

    compute_and_save_avg_scores(
        model=model,
        train_path=args.train_path,
        val_emb=val_emb,
        encode_batch_size=args.encode_batch_size,
        text_max_chars=args.text_max_chars,
        topk=args.topk,
        avg_scores_output_path=args.avg_scores_output_path,
        checkpoint_path=args.checkpoint_path,
        checkpoint_every_batches=args.checkpoint_every_batches,
        resume=args.resume,
    )

    print("Done.")
    print(f"Avg score file: {args.avg_scores_output_path}")
    print(f"Checkpoint: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
