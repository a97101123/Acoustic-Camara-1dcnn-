from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .run_io import load_config

_YAML_SCHEMAS = {
    "run-workflow": ({"steps"}, "一鍵流程"),
    "build-features": ({"process", "input", "output"}, "建立特徵"),
    "train-batch": ({"selection", "training", "output"}, "批次訓練"),
}


def validate_yaml_command(config_path: str | Path, command: str) -> tuple[bool, str]:
    """Return an actionable GUI message when a YAML belongs to another command."""
    try:
        _, cfg = load_config(config_path)
    except Exception as exc:
        return False, f"無法讀取 YAML：{exc}"
    if not isinstance(cfg, dict):
        return False, "YAML 最上層必須是欄位表。"
    if command not in _YAML_SCHEMAS:
        return False, f"不支援命令：{command}"
    required, label = _YAML_SCHEMAS[command]
    missing = sorted(required - set(cfg))
    if not missing:
        return True, ""
    detected = []
    for _, (keys, name) in _YAML_SCHEMAS.items():
        if keys.issubset(cfg):
            detected.append(name)
    hint = f" 此檔案看起來是「{detected[0]}」設定，請按「{detected[0]}」。" if detected else ""
    return False, f"「{label}」設定缺少：{', '.join(missing)}。{hint}"


def main() -> None:
    try:
        from PyQt6.QtCore import QThread
        from PyQt6.QtCore import pyqtSignal as Signal
        from PyQt6.QtWidgets import (
            QApplication,
            QComboBox,
            QFileDialog,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QPlainTextEdit,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError('GUI requires PyQt6: python -m pip install -e ".[gui]"') from exc

    class Worker(QThread):
        output = Signal(str)
        done = Signal(int)

        def __init__(self, args):
            super().__init__()
            self.args = args

        def run(self):
            process = subprocess.Popen(
                [sys.executable, "-m", "wear_cnn.cli", *self.args],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in process.stdout:
                self.output.emit(line.rstrip())
            self.done.emit(process.wait())

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Wear CNN")
            self.resize(900, 760)
            box = QVBoxLayout()
            row = QHBoxLayout()
            self.config = QLineEdit()
            browse = QPushButton("選擇 YAML")
            browse.clicked.connect(self.pick)
            row.addWidget(QLabel("設定"))
            row.addWidget(self.config)
            row.addWidget(browse)
            box.addLayout(row)
            grid = QGridLayout()
            self.fields = {}
            names = [
                "features",
                "resume",
                "run",
                "deployment",
                "mic_wav",
                "acc_x",
                "acc_y",
                "acc_z",
            ]
            directories = {"features", "resume", "run"}
            yaml_fields = {"deployment"}
            for index, name in enumerate(names):
                field = QLineEdit()
                self.fields[name] = field
                grid.addWidget(QLabel(name), index, 0)
                grid.addWidget(field, index, 1)
                if name != "group":
                    button = QPushButton("瀏覽")
                    button.clicked.connect(
                        lambda checked=False, n=name, d=name in directories, y=name in yaml_fields: (
                            self.pick_field(n, d, y)
                        )
                    )
                    grid.addWidget(button, index, 2)
            self.device = QComboBox()
            self.device.addItems(["auto", "cpu", "cuda"])
            grid.addWidget(QLabel("device"), len(names), 0)
            grid.addWidget(self.device, len(names), 1)
            box.addLayout(grid)
            buttons = QGridLayout()
            self.command_buttons = []
            commands = [
                ("一鍵流程", "run-workflow"),
                ("建立特徵", "build-features"),
                ("批次訓練", "train-batch"),
                ("重建報告", "render-training-report"),
                ("MIC推論", "predict"),
                ("MIC/ACC推論", "predict-fusion"),
            ]
            for index, (label, command) in enumerate(commands):
                button = QPushButton(label)
                button.clicked.connect(lambda checked=False, c=command: self.run_command(c))
                buttons.addWidget(button, index // 3, index % 3)
                self.command_buttons.append(button)
            box.addLayout(buttons)
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            box.addWidget(self.log)
            root = QWidget()
            root.setLayout(box)
            self.setCentralWidget(root)
            self.worker = None

        def pick(self):
            path, _ = QFileDialog.getOpenFileName(self, "設定檔", "", "YAML (*.yaml *.yml)")
            if path:
                self.config.setText(path)

        def pick_field(self, name, directory=False, json_file=False):
            if directory:
                path = QFileDialog.getExistingDirectory(self, "選擇資料夾")
            else:
                file_filter = (
                    "JSON (*.json)"
                    if json_file
                    else "資料檔 (*.h5 *.hdf5 *.pt *.wav *.xlsx);;所有檔案 (*)"
                )
                path, _ = QFileDialog.getOpenFileName(self, "選擇檔案", "", file_filter)
            if path:
                self.fields[name].setText(path)

        def run_command(self, command):
            args = [command]
            if command in _YAML_SCHEMAS:
                if not self.config.text():
                    self.log.appendPlainText("缺少設定 YAML")
                    return
                valid, message = validate_yaml_command(self.config.text(), command)
                if not valid:
                    self.log.appendPlainText(message)
                    return
                args += ["--config", self.config.text()]
                if command == "train-batch":
                    for field in ("features", "resume"):
                        if self.fields[field].text().strip():
                            args += ["--" + field, self.fields[field].text().strip()]
            elif command == "render-training-report":
                args += ["--run", self.fields["run"].text()]
            elif command == "predict":
                args += [
                    "--deployment",
                    self.fields["deployment"].text(),
                    "--wav",
                    self.fields["mic_wav"].text(),
                    "--device",
                    self.device.currentText(),
                ]
            elif command == "predict-fusion":
                args += [
                    "--deployment",
                    self.fields["deployment"].text(),
                    "--mic-wav",
                    self.fields["mic_wav"].text(),
                    "--acc-wav",
                    self.fields["acc_x"].text(),
                    self.fields["acc_y"].text(),
                    self.fields["acc_z"].text(),
                    "--device",
                    self.device.currentText(),
                ]
            required = {
                "render-training-report": ["run"],
                "predict": ["deployment", "mic_wav"],
                "predict-fusion": ["deployment", "mic_wav", "acc_x", "acc_y", "acc_z"],
            }.get(command, [])
            missing = [name for name in required if not self.fields[name].text().strip()]
            if missing:
                self.log.appendPlainText("缺少欄位: " + ", ".join(missing))
                return
            self.log.appendPlainText("> " + " ".join(args))
            self.worker = Worker(args)
            self.worker.output.connect(self.log.appendPlainText)
            for button in self.command_buttons:
                button.setEnabled(False)

            def finished(code):
                self.log.appendPlainText(f"完成 code={code}")
                for button in self.command_buttons:
                    button.setEnabled(True)

            self.worker.done.connect(finished)
            self.worker.start()

    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
