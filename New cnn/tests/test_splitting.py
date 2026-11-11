import pandas as pd
import settings as explicit

from wear_cnn.splitting import apply_labels, split_tools, training_tertiles


def test_tool_split_is_isolated_and_repeatable():
    frame = pd.DataFrame(
        {
            "ToolID": [f"T{i}" for i in range(10) for _ in range(2)],
            "RecordID": range(20),
            "WearValue": range(20),
        }
    )
    a = split_tools(frame, 42, explicit.evaluation())
    b = split_tools(frame, 42, explicit.evaluation())
    assert a.Split.tolist() == b.Split.tolist()
    assert all(a.groupby("ToolID").Split.nunique() == 1)
    assert {"train", "val", "test"} == set(a.Split)


def test_thresholds_only_use_training():
    frame = pd.DataFrame({"Split": ["train"] * 3 + ["test"], "WearValue": [0, 1, 2, 999]})
    thresholds = training_tertiles(frame)
    assert thresholds[1] < 999
    assert apply_labels(frame, thresholds).Label.iloc[-1] == 2
