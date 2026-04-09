import argparse
import heapq
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from zhipuai import ZhipuAI
import tiktoken


def build_text_for_embedding(record: Dict) -> str:
    """Build embedding text for different schema styles."""
    if "feature_content" in record and str(record["feature_content"]).strip():
        return str(record["feature_content"]).strip()

    # Common SFT style: instruction/input/output
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

    # Common QA/MCQ style
    if "question" in record:
        parts = [f"question: {record.get('question', '')}"]
        for key in ["A", "B", "C", "D", "E", "answer", "explanation"]:
            if key in record and str(record.get(key, "")).strip():
                parts.append(f"{key}: {record.get(key)}")
        return "\n".join(parts)

    # Fallback: concatenate scalar fields
    parts = []
    for k, v in record.items():
        if k in {"feature_vector", "matches", "avg_score"}:
            continue
        if isinstance(v, (str, int, float, bool)):
            text = str(v).strip()
            if text:
                parts.append(f"{k}: {text}")
    return "\n".join(parts)

def truncate_by_tokens(text: str, max_tokens: int = 3000) -> str:
    """将文本截断到不超过 max_tokens 个 token（使用 cl100k_base 编码，与 embedding-3 一致）"""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return enc.decode(truncated_tokens)

def get_embedding(client: ZhipuAI, text: str, model: str, max_retries: int = 3) -> List[float]:
    # for attempt in range(max_retries):
    #     try:
    #         resp = client.embeddings.create(model=model, input=text)
    #         return resp.data[0].embedding
    #     except Exception:
    #         if attempt == max_retries - 1:
    #             raise
    #         time.sleep(2 ** attempt)
    # raise RuntimeError("Unreachable")
    original_len = len(text)
    text = truncate_by_tokens(text, 3072)
    if len(text) != original_len:
        print(f"Truncated text from {original_len} chars to {len(text)} chars (max_tokens={3072})")

    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=text)
            return resp.data[0].embedding
        except Exception as e:
            msg = str(e)
            if "400" in msg or "1210" in msg:
                raise
            print(f"Attempt {attempt+1} failed. Text length: {len(text)}")
            print(f"Text preview: {text[:200]}")
            print(f"Error: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


def vectorize_val(
    client: ZhipuAI,
    val_path: str,
    model: str,
    max_workers: int,
    save_val_vectors_path: str,
) -> Tuple[List[Dict], np.ndarray]:
    with open(val_path, "r", encoding="utf-8") as f:
        val_rows = [json.loads(line) for line in f if line.strip()]

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, row in enumerate(val_rows):
            text = build_text_for_embedding(row)
            tasks.append((idx, ex.submit(get_embedding, client, text, model)))

        for idx, fut in tqdm(tasks, desc="Vectorizing val"):
            val_rows[idx]["feature_vector"] = fut.result()
            if "id" not in val_rows[idx]:
                val_rows[idx]["id"] = idx

    with open(save_val_vectors_path, "w", encoding="utf-8") as f:
        for row in val_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    val_matrix = np.array([r["feature_vector"] for r in val_rows], dtype=np.float32)
    val_norm = np.linalg.norm(val_matrix, axis=1, keepdims=True)
    val_norm = np.where(val_norm == 0, 1e-10, val_norm)
    val_matrix = val_matrix / val_norm
    return val_rows, val_matrix


def topk_matches(
    train_vec: np.ndarray,
    val_rows: List[Dict],
    val_norm_matrix: np.ndarray,
    topk: int,
) -> Tuple[List[Dict], float]:
    train_norm = np.linalg.norm(train_vec)
    if train_norm == 0:
        train_norm = 1e-10
    sims = np.dot(val_norm_matrix, train_vec / train_norm)

    top_idx = np.argpartition(-sims, topk - 1)[:topk]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    matches = []
    for idx in top_idx:
        item = dict(val_rows[int(idx)])
        item.pop("feature_vector", None)
        item["match_score"] = float(sims[int(idx)])
        matches.append(item)

    avg_score = float(np.mean([m["match_score"] for m in matches])) if matches else 0.0
    return matches, avg_score


def wait_any(futures_set):
    done, not_done = wait(futures_set, return_when=FIRST_COMPLETED)
    return done, not_done


def process_train(
    client: ZhipuAI,
    train_path: str,
    model: str,
    max_workers: int,
    topk: int,
    final_topn: int,
    val_rows: List[Dict],
    val_norm_matrix: np.ndarray,
    output_topn_path: str,
    save_train_vectors_path: str = "",
):
    do_save_train_vectors = bool(save_train_vectors_path.strip())
    f_train_vec = open(save_train_vectors_path, "w", encoding="utf-8") if do_save_train_vectors else None

    heap: List[Tuple[float, int, Dict]] = []
    counter = 0

    def embed_one(i_line: Tuple[int, str]):
        idx, line = i_line
        row = json.loads(line)
        text = build_text_for_embedding(row)
        vec = get_embedding(client, text, model)
        if "id" not in row:
            row["id"] = idx
        return idx, row, vec

    with open(train_path, "r", encoding="utf-8") as f_count:
        total = sum(1 for line in f_count if line.strip())

    with ThreadPoolExecutor(max_workers=max_workers) as ex, open(train_path, "r", encoding="utf-8") as f:
        in_flight = set()
        max_in_flight = max_workers * 4
        line_iter = ((i, line) for i, line in enumerate(f) if line.strip())
        progress = tqdm(total=total, desc="Vectorizing+matching train")

        while True:
            while len(in_flight) < max_in_flight:
                try:
                    item = next(line_iter)
                except StopIteration:
                    break
                in_flight.add(ex.submit(embed_one, item))

            if not in_flight:
                break

            done, in_flight = wait_any(in_flight)
            for fut in done:
                idx, row, vec = fut.result()
                row["feature_vector"] = vec

                if do_save_train_vectors and f_train_vec:
                    f_train_vec.write(json.dumps(row, ensure_ascii=False) + "\n")

                train_vec = np.array(vec, dtype=np.float32)
                matches, avg_score = topk_matches(train_vec, val_rows, val_norm_matrix, topk=topk)

                row.pop("feature_vector", None)
                row["matches"] = matches
                row["avg_score"] = avg_score

                if len(heap) < final_topn:
                    heapq.heappush(heap, (avg_score, counter, row))
                else:
                    if avg_score > heap[0][0]:
                        heapq.heapreplace(heap, (avg_score, counter, row))
                counter += 1
                progress.update(1)

        progress.close()

    if f_train_vec:
        f_train_vec.close()

    selected = [x[2] for x in sorted(heap, key=lambda t: t[0], reverse=True)]
    with open(output_topn_path, "w", encoding="utf-8") as fout:
        for item in selected:
            item.pop("avg_score", None)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Vectorize train/val, match train->val top5, rank by top5 avg score, export topN."
    )
    parser.add_argument("--train_path", default="train.jsonl")
    parser.add_argument("--val_path", default="val.jsonl")
    parser.add_argument("--api_key", required=True, help="Zhipu API key")
    parser.add_argument("--embedding_model", default="embedding-3")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--final_topn", type=int, default=100000)
    parser.add_argument("--save_val_vectors_path", default="val_with_vectors.jsonl")
    parser.add_argument(
        "--save_train_vectors_path",
        default="",
        help="Optional. If provided, save full vectorized train records to this path.",
    )
    parser.add_argument("--output_topn_path", default="train_top100k_by_val_similarity.jsonl")
    args = parser.parse_args()

    client = ZhipuAI(api_key=args.api_key)

    print("Step 1/3: vectorizing val...")
    val_rows, val_norm_matrix = vectorize_val(
        client=client,
        val_path=args.val_path,
        model=args.embedding_model,
        max_workers=args.max_workers,
        save_val_vectors_path=args.save_val_vectors_path,
    )

    print("Step 2/3: vectorizing train and matching train->val...")
    process_train(
        client=client,
        train_path=args.train_path,
        model=args.embedding_model,
        max_workers=args.max_workers,
        topk=args.topk,
        final_topn=args.final_topn,
        val_rows=val_rows,
        val_norm_matrix=val_norm_matrix,
        output_topn_path=args.output_topn_path,
        save_train_vectors_path=args.save_train_vectors_path,
    )

    print("Step 3/3: done.")
    print(f"Top-N output: {args.output_topn_path}")
    print(f"Val vectors: {args.save_val_vectors_path}")
    if args.save_train_vectors_path:
        print(f"Train vectors: {args.save_train_vectors_path}")


if __name__ == "__main__":
    main()
