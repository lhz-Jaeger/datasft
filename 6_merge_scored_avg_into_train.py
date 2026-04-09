import argparse
import json
import os

from tqdm import tqdm


def count_nonempty_lines(path: str) -> int:
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def load_avg_map(scored_path: str):
    avg_map = {}
    total = count_nonempty_lines(scored_path)
    with open(scored_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc="Loading scored avg"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            row_id = item.get("id")
            if row_id is None:
                continue
            avg_map[int(row_id)] = float(item.get("avg_score", 0.0))
    return avg_map


def merge_avg_into_train(train_path: str, avg_map: dict, output_path: str):
    total = count_nonempty_lines(train_path)
    hit = 0

    with open(train_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for idx, line in tqdm(enumerate(fin), total=total, desc="Merging avg into train"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if idx in avg_map:
                row["avg_score"] = avg_map[idx]
                hit += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Merged rows: {hit}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge avg_score from scored file back into train.jsonl.")
    parser.add_argument("--train_path", default="train.jsonl")
    parser.add_argument("--scored_path", default="train_scored_bge_m3.jsonl")
    parser.add_argument("--output_path", default="train.jsonl.merged")
    parser.add_argument("--in_place", action="store_true", help="Replace train_path with output_path after merging.")
    args = parser.parse_args()

    avg_map = load_avg_map(args.scored_path)
    merge_avg_into_train(args.train_path, avg_map, args.output_path)

    if args.in_place:
        os.replace(args.output_path, args.train_path)
        print(f"Replaced {args.train_path} in place.")


if __name__ == "__main__":
    main()
