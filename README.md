# DP2020
Source code for my diploma thesis written in 2020.

## _Disclaimer:_
In retrospect, even for me today many things in the code seem outdated or impractical. The repository is pretty bare and it lacks in terms of best coding pracitces, structuring and automatization.

I went through the code and compiled a (not exhaustive) list of improvements which I would set up and implement if I wrote this today.  

Sorted by the priority in a descending order, those are:

- use formatter tools like `black`, `isort` etc.
- use typehints and static typechecker system like `mypy`.
- I'd probably spend more time setting up a `pre-commit` routine to run the formatting and typechecks automatically.
- better structured docstrings which would allow for generation of HTML documentation using `Sphinx` for example.
- write unit tests since day one (omitted back then due to the lack of time).
- use some sophisticated dependency manager (like `Poetry`).
- use convenience libraries like `pathlib`, `typer`/`click` etc. I'd think more about the `logging` hierarchy. 
- I'd consider using `Pytorch Lightning` to reduce boilerplate code and to get access to some additional goods (the callback system, robust logging, useful classes etc.).
- better configuration loading and processing. Today, I'd use `Hydra` (which didn't exists back then) or some simpler tool.
- I'd add a `Dockerfile` if there was time left.
- maybe I'd set up some very simple MLOps pipeline including model repository (`MLFlow`, `ClearML`) to be able to store experiments and a CI pipeline to test new code.
