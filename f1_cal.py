import json
import argparse

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_finish_reason(record):
    try:
        fr = record["response"]["choices"][0]["finish_reason"]
        if fr is not None:
            return fr
    except Exception:
        pass
    return record.get("finish_reason")

def is_tool_call(record):
    return get_finish_reason(record) == "tool_calls"

def main(model_path, official_path, output_json):
    model_data = load_jsonl(model_path)
    official_data = load_jsonl(official_path)

    if len(model_data) != len(official_data):
        print(
            f"WARNING: model({len(model_data)}) != official({len(official_data)})"
        )

    TP = FP = FN = TN = 0
    length = min(len(model_data), len(official_data))

    for i in range(length):
        model_tc = is_tool_call(model_data[i])
        official_tc = is_tool_call(official_data[i])

        if model_tc and official_tc:
            TP += 1
        elif model_tc and not official_tc:
            FP += 1
        elif not model_tc and official_tc:
            FN += 1
        else:
            TN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    result = {
        "counts": {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "total": length
        },
        "metrics": {
            "tool_call_precision": round(precision, 6),
            "tool_call_recall": round(recall, 6),
            "model_official_similarity": round(f1, 6)
        },
        "definition": {
            "positive_label": "finish_reason == 'tool_calls'",
            "ground_truth": "official_api",
            "source": "response.choices[0].finish_reason"
        }
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--official", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    main(args.model, args.official, args.output_json)
