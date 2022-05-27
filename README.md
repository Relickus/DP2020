# DP2020
Source code for my diploma thesis written (without a professional programming experience) in 2020.

## _Disclaimer:_
In hindsight, many things in the code seem outdated or impractical. The repository is pretty bare and it lacks in terms of best coding practices, structuring and automatization. I intentionally left the repository in this state to have a threshold to measure the shift in my programming skills. 

I went through the code and compiled a (not exhaustive) list of improvements which I would implement if I wrote this today.

Sorted by the priority in a descending order, those are:

- use formatter tools like `black`, `isort` etc.
- use typehints and static typechecker system like `mypy`.
- I'd set up a `pre-commit` routine to run the aforementioned checks/formats (and a few others) automatically.
- I'd sacrifice some performance and use `pydantic` or `attrs` for automatic data validation.
- I'd set up a simple CI pipeline to test new code using Github actions or a similar tool.
- better structured docstrings which would allow for generation of HTML documentation using `Sphinx` for example.
- write unit tests since day one (omitted back then due to the lack of time) and pursuit some reasonable coverage level.
- use some sophisticated dependency manager (like `Poetry`).
- use convenience libraries like `pathlib`, `typer`/`click`, `loguru` etc.
- I'd consider using `Pytorch Lightning` to reduce boilerplate code and to get access to some additional goods (the callback system, robust logging, useful classes etc.).
- better configuration loading and processing. Today, I'd use `Hydra` (which didn't exists back then) or some simpler tool like `Dynaconf`.
- I'd add a `Dockerfile` if there was time left.
- I'd probably set up some simple MLOps pipeline using `MLFlow` or `ClearML` to be able to store and compare models (it gets messy with just tensorboard and local file system storage)
