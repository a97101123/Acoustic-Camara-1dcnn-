# Python environments

Use .venv-cuda/Scripts/python.exe for GPU execution and .venv-review/Scripts/python.exe for CPU tests. Launch scripts prefer the existing CUDA environment, then the review environment; -PythonPath overrides this selection.

```powershell
.\.venv-cuda\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
.\tools\run_architecture.ps1 -Process turning -Stage features -ValidateOnly
.\tools\run_four_architecture.ps1
```

For another machine, create a Python environment and install the project with pip install -e ".[dev,train]". Install the gui extra to run the optional Qt GUI. Change input.data_root and regenerate current source tables for that machine.

Do not run concurrent trainers against the same output directory. Use a fresh Feature Set after preprocessing changes. Existing old results are not accepted by this version.
