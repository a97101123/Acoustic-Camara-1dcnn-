import matplotlib.pyplot as plt
import pytest

from wear_cnn.features import plots


@pytest.mark.parametrize(
    "method,threshold,line_count",
    [
        ("r2", None, 0),
        ("r2", 0.0, 1),
        ("r2", 0.5, 1),
        ("j", None, 0),
    ],
)
def test_score_plot_supports_top_and_threshold_modes(
    tmp_path, monkeypatch, method, threshold, line_count
):
    fig, ax = plt.subplots()
    monkeypatch.setattr(plots.plt, "subplots", lambda **kwargs: (fig, ax))
    output = tmp_path / "scores.png"
    plots.plot_score_overview(["a", "b"], [0.1, 0.3], [1], method, output, 40, threshold=threshold)
    assert output.read_bytes().startswith(b"\x89PNG")
    assert len(ax.lines) == line_count
    if threshold is not None:
        assert list(ax.lines[0].get_ydata()) == [threshold, threshold]
