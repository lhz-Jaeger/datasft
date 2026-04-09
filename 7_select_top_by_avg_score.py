import argparse
import heapq
import json
from typing import Dict, List, Set, Tuple

from tqdm import tqdm


def count_nonempty_lines(path: str) -> int:
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def load_top_ids(avg_scores_path: str, topn: int) -> Set[int]:
    total = count_nonempty_lines(avg_scores_path)
    heap: List[Tuple[float, int]] = []

    with open(avg_scores_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc="Selecting top ids by avg_score"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            row_id = item.get("id", None)
            avg = item.get("avg_score", None)
            if row_id is None or avg is None:
                continue
            row_id = int(row_id)
            avg = float(avg)

            if len(heap) < topn:
                heapq.heappush(heap, (avg, row_id))
            else:
                if avg > heap[0][0]:
                    heapq.heapreplace(heap, (avg, row_id))

    return {row_id for _, row_id in heap}


def extract_rows_by_ids(
    train_path: str,
    selected_ids: Set[int],
    output_path: str,
    keep_avg_score: bool,
):
    total = count_nonempty_lines(train_path)
    written = 0

    with open(train_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for raw_idx, line in tqdm(enumerate(fin), total=total, desc="Extracting rows from train"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = int(row.get("id", raw_idx))
            if row_id not in selected_ids:
                continue
            if not keep_avg_score:
                row.pop("avg_score", None)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved rows: {written}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Select top-N by avg_score file, then extract full records from train.jsonl."
    )
    parser.add_argument("--train_path", default="train.jsonl")
    parser.add_argument("--avg_scores_path", default="train_avg_scores_bge_m3.jsonl")
    parser.add_argument("--output_path", default="train_top100k_by_avg_score.jsonl")
    parser.add_argument("--topn", type=int, default=100000)
    parser.add_argument("--keep_avg_score", action="store_true")
    args = parser.parse_args()

    selected_ids = load_top_ids(args.avg_scores_path, args.topn)
    extract_rows_by_ids(
        train_path=args.train_path,
        selected_ids=selected_ids,
        output_path=args.output_path,
        keep_avg_score=args.keep_avg_score,
    )


if __name__ == "__main__":
    main()
