from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from check import build_submission_check, write_submission_check


REPORT_COLUMNS = ("feature", "value")
EXPECTED_SAMPLE_COUNT = 803


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Finalize processed BIG 2015 submission")
    parser.add_argument("--project", type=Path, default=project_root)
    parser.add_argument(
        "--source-zip",
        type=Path,
        default=project_root.parent / "kaggle2015-sample" / "subtrain.zip",
    )
    parser.add_argument(
        "--source-labels",
        type=Path,
        default=project_root.parent / "kaggle2015-sample" / "subtrainLabels.csv",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quality_level(unknown_ratio: float, head_unknown_ratio: float) -> tuple[str, str]:
    if unknown_ratio > 0.90 or head_unknown_ratio > 0.50:
        reasons = []
        if unknown_ratio > 0.90:
            reasons.append("unknown>90%")
        if head_unknown_ratio > 0.50:
            reasons.append("head_unknown>50%")
        return "review", ";".join(reasons)
    if unknown_ratio > 0.50:
        return "caution", "unknown>50%"
    return "normal", ""


def require_full_outputs(data: Path, manifests: Path, reports: Path) -> None:
    required = [
        manifests / "image_records.csv",
        reports / "image_summary.json",
        data / "images",
        data / "images_head",
        data / "masks",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required full-file outputs: " + ", ".join(missing))


def ensure_head_image_hashes(
    data: Path,
    manifests: Path,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    frame = manifest.copy()
    changed = False
    if "image_file_sha256" not in frame.columns:
        insert_at = frame.columns.get_loc("image_path") + 1
        frame.insert(insert_at, "image_file_sha256", "")
        changed = True
    missing = frame["image_file_sha256"].isna() | frame["image_file_sha256"].eq("")
    for index, row in frame.loc[missing].iterrows():
        image_path = data / str(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing 32x32 head image: {image_path}")
        frame.at[index, "image_file_sha256"] = sha256_file(image_path)
        changed = True
    if changed:
        frame.to_csv(manifests / "samples.csv", index=False, encoding="utf-8-sig")
    return frame


def build_constant_report(
    features: pd.DataFrame,
    report_path: Path,
) -> tuple[list[str], pd.DataFrame]:
    constant_columns = [
        column
        for column in features.columns
        if column not in {"Id", "Class"} and features[column].nunique(dropna=False) <= 1
    ]
    current = pd.DataFrame(
        {
            "feature": constant_columns,
            "value": [features[column].iloc[0] for column in constant_columns],
        }
    )
    retained = pd.DataFrame(columns=REPORT_COLUMNS)
    if report_path.is_file():
        try:
            previous = pd.read_csv(report_path)
        except pd.errors.EmptyDataError:
            previous = pd.DataFrame(columns=REPORT_COLUMNS)
        if set(REPORT_COLUMNS).issubset(previous.columns):
            retained = previous.loc[
                ~previous["feature"].astype(str).isin(features.columns),
                list(REPORT_COLUMNS),
            ]
    report = pd.concat([retained, current], ignore_index=True)
    if not report.empty:
        report = (
            report.drop_duplicates(subset="feature", keep="last")
            .sort_values("feature")
            .reset_index(drop=True)
        )
    return constant_columns, report


def build_quality_flags(manifest: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    quality = manifest[["Id", "Class", "split", "unknown_byte_ratio"]].merge(
        features[["Id", "image_unknown_tokens"]],
        on="Id",
        how="left",
        validate="one_to_one",
    )
    quality["head_unknown_ratio"] = quality["image_unknown_tokens"] / 1024.0
    levels = quality.apply(
        lambda row: quality_level(
            float(row["unknown_byte_ratio"]),
            float(row["head_unknown_ratio"]),
        ),
        axis=1,
    )
    quality["quality"] = [item[0] for item in levels]
    quality["reason"] = [item[1] for item in levels]
    quality = quality.rename(columns={"unknown_byte_ratio": "unknown_ratio"})
    quality["unknown_ratio"] = quality["unknown_ratio"].round(6)
    quality["head_unknown_ratio"] = quality["head_unknown_ratio"].round(6)
    return quality[
        ["Id", "Class", "split", "unknown_ratio", "head_unknown_ratio", "quality", "reason"]
    ]


def build_dataset(
    samples: pd.DataFrame,
    image_records: pd.DataFrame,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    sample_ids = set(samples["Id"].astype(str))
    if sample_ids != set(image_records["Id"].astype(str)):
        raise ValueError("IDs differ between samples.csv and image_records.csv")
    if sample_ids != set(quality["Id"].astype(str)):
        raise ValueError("IDs differ between samples.csv and quality.csv")
    dataset = samples[["Id", "Class", "split", "image_path"]].merge(
        image_records[["Id", "full_image_path", "full_unknown_mask_path"]],
        on="Id",
        how="inner",
        validate="one_to_one",
    ).merge(
        quality[["Id", "unknown_ratio", "quality"]],
        on="Id",
        how="inner",
        validate="one_to_one",
    )
    dataset = dataset.rename(
        columns={
            "full_image_path": "image",
            "image_path": "head_image",
            "full_unknown_mask_path": "mask",
        }
    )
    columns = ["Id", "Class", "split", "image", "head_image", "mask", "unknown_ratio", "quality"]
    dataset = dataset[columns]
    prefixes = {"image": "images/", "head_image": "images_head/", "mask": "masks/"}
    for column, prefix in prefixes.items():
        dataset[column] = dataset[column].astype(str).str.replace("\\", "/", regex=False)
        invalid = ~dataset[column].str.startswith(prefix)
        if invalid.any():
            raise ValueError(f"{column} paths must start with {prefix}")
    return dataset


def update_processing_summary(
    summary_path: Path,
    full_summary_path: Path,
    checks: dict[str, object],
    constant_report: pd.DataFrame,
) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    full_summary = json.loads(full_summary_path.read_text(encoding="utf-8"))
    if isinstance(summary.get("input"), dict) and summary["input"].get("zip_path"):
        summary["input"]["zip_path"] = Path(str(summary["input"]["zip_path"])).name
    for key in ("verification", "full_image_generation", "submission_check"):
        summary.pop(key, None)
    output = summary.setdefault("output", {})
    for key in ("image_representations", "image_size", "image_mode"):
        output.pop(key, None)
    output["feature_columns"] = checks["feature_count_excluding_id_and_label"]
    output["images"] = {
        "head": {
            "directory": "images_head",
            "count": checks["head_images"]["found"],
            "size": [32, 32],
            "mode": "L",
        },
        "main": {
            "directory": "images",
            "count": checks["images"]["found"],
            "size": [32, 32],
            "mode": "L",
        },
        "mask": {
            "directory": "masks",
            "count": checks["masks"]["found"],
            "size": [32, 32],
            "mode": "L",
        },
    }
    summary["image_generation"] = {
        "method": full_summary.get("method"),
        "method_version": full_summary.get("method_version"),
        "summary_path": "data/reports/image_summary.json",
        "passed": bool(full_summary.get("verification", {}).get("passed")),
        "verified_samples": int(
            full_summary.get("verification", {}).get("verified_samples", 0)
        ),
    }
    summary["check"] = {
        "passed": checks["passed"],
        "removed_features": int(len(constant_report)),
        "quality_counts": checks["quality_counts"],
        "head_count": checks["head_images"]["found"],
        "image_count": checks["images"]["found"],
        "mask_count": checks["masks"]["found"],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    data = project / "data"
    manifests = data / "manifests"
    reports = data / "reports"
    features_path = data / "features" / "features.csv"
    manifest_path = manifests / "samples.csv"
    full_manifest_path = manifests / "image_records.csv"
    summary_path = reports / "summary.json"
    full_summary_path = reports / "image_summary.json"
    require_full_outputs(data, manifests, reports)
    features = pd.read_csv(features_path, dtype={"Id": str})
    manifest = pd.read_csv(manifest_path, dtype={"Id": str})
    full_manifest = pd.read_csv(full_manifest_path, dtype={"Id": str})
    manifest = ensure_head_image_hashes(data, manifests, manifest)
    if len(manifest) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"Expected {EXPECTED_SAMPLE_COUNT} samples, found {len(manifest)}")
    if not (
        set(features["Id"].astype(str))
        == set(manifest["Id"].astype(str))
        == set(full_manifest["Id"].astype(str))
    ):
        raise ValueError("IDs differ among features.csv, samples.csv, and image_records.csv")
    constant_report_path = manifests / "removed_features.csv"
    constant_columns, constant_report = build_constant_report(features, constant_report_path)
    if constant_columns:
        features = features.drop(columns=constant_columns)
        features.to_csv(features_path, index=False, encoding="utf-8-sig")
    constant_report.to_csv(constant_report_path, index=False, encoding="utf-8-sig")
    quality = build_quality_flags(manifest, features)
    quality.to_csv(manifests / "quality.csv", index=False, encoding="utf-8-sig")
    dataset = build_dataset(manifest, full_manifest, quality)
    dataset.to_csv(data / "dataset.csv", index=False, encoding="utf-8-sig")
    checks = build_submission_check(
        project=project,
        source_zip=args.source_zip.resolve(),
        source_labels=args.source_labels.resolve(),
    )
    write_submission_check(project, checks)
    update_processing_summary(summary_path, full_summary_path, checks, constant_report)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
