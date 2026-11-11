from __future__ import annotations

import argparse

CURRENT_WORKFLOW = "Current workflow: build-features then train-batch."


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI without importing training or plotting dependencies."""
    parser = argparse.ArgumentParser(prog="wear-cnn", description=CURRENT_WORKFLOW)
    subparsers = parser.add_subparsers(dest="command", required=True)
    builder = subparsers.add_parser("build-features")
    builder.add_argument("--config", required=True)
    builder.add_argument("--validate", action="store_true")
    trainer = subparsers.add_parser("train-batch")
    trainer.add_argument("--config", required=True)
    trainer.add_argument("--features")
    trainer.add_argument("--resume")
    report_run = subparsers.add_parser("render-training-report")
    report_run.add_argument("--run", required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--deployment", required=True)
    predict.add_argument("--wav", required=True, nargs="+")
    predict.add_argument("--device", required=True, choices=["auto", "cpu", "cuda"])
    fusion_predict = subparsers.add_parser("predict-fusion")
    fusion_predict.add_argument("--deployment", required=True)
    fusion_predict.add_argument("--mic-wav", required=True)
    fusion_predict.add_argument("--acc-wav", required=True, nargs=3)
    fusion_predict.add_argument("--mic-start", type=float, help="MIC cut start in seconds")
    fusion_predict.add_argument("--acc-start", type=float, help="Shared ACC XYZ cut start in seconds")
    fusion_predict.add_argument("--duration", type=float, help="Shared cut duration in seconds")
    fusion_predict.add_argument("--device", required=True, choices=["auto", "cpu", "cuda"])
    workflow = subparsers.add_parser("run-workflow")
    workflow.add_argument("--config", required=True)
    return parser


def run_command(args: argparse.Namespace):
    """Import and execute only the selected command; leave printing to main."""
    if args.command == "build-features":
        from .feature_builder import build_features, validate_feature_config

        if args.validate:
            _, cfg, frame, _, nffts = validate_feature_config(args.config)
            return f"Validated {cfg['process']}: {len(frame)} records, {len(nffts)} NFFTs"
        else:
            return build_features(args.config)
    elif args.command == "train-batch":
        from .trainer_batch import train_batch

        return train_batch(args.config, args.features, args.resume)
    elif args.command == "render-training-report":
        from .trainer_batch import render_training_report

        return render_training_report(args.run)
    elif args.command == "predict":
        import json

        from wear_cnn.inference import predict_wav

        return json.dumps(
            predict_wav(args.wav, args.deployment, args.device), ensure_ascii=False, indent=2
        )
    elif args.command == "predict-fusion":
        import json

        from wear_cnn.inference import predict_fusion

        return json.dumps(
            predict_fusion(
                args.mic_wav, args.acc_wav, args.deployment, args.device,
                mic_start=args.mic_start, acc_start=args.acc_start, duration=args.duration,
            ),
            ensure_ascii=False,
            indent=2,
        )
    elif args.command == "run-workflow":
        from wear_cnn.workflow import run_workflow

        return run_workflow(args.config)
    raise ValueError("Unknown CLI command")


def main() -> None:
    args = build_parser().parse_args()
    result = run_command(args)
    print(result)
    if args.command == "train-batch":
        import json
        from pathlib import Path

        summary = json.loads((Path(result) / "run_summary.json").read_text(encoding="utf-8"))
        if summary["failed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
