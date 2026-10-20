# Contributing

## Reproducing the CI test environment

Regular CI uses the committed `uv.lock`, while retaining Hatch to create and run
test environments. The lock covers all project extras, Hatch's test plugins,
and the bootstrap and editable-build dependencies in the `ci` dependency group.
Python and platform markers select the appropriate versions for each CI runner.

Use **uv 0.12.1**, matching `.github/workflows/testing.yml`. In a fresh checkout,
the following Bash commands reproduce the Python 3.12 test environment:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv export --locked --all-extras --group ci --no-emit-project --no-hashes \
  --output-file .venv/ci-constraints.txt
export UV_CONSTRAINT="$(python -c 'from pathlib import Path; print(Path(".venv/ci-constraints.txt").resolve())')"
export UV_BUILD_CONSTRAINT="$UV_CONSTRAINT"
uv pip install hatch
hatch env remove hatch-test.py3.12
hatch env create hatch-test.py3.12
uv pip list --python "$(hatch env find hatch-test.py3.12)"
hatch test --python 3.12
```

On Windows, use Git Bash and replace the activation line with
`source .venv/Scripts/activate`. Replace `3.12` consistently to test another
supported Python version. Removing the Hatch environment discards its installed
packages so the next creation starts cleanly; it does not remove project files.
To check repeatability, recreate it again and compare the package/version lists
on the same Python version and operating system.

The two constraint variables apply to dependency installation and isolated
builds, respectively. CI exports them only for the regular test jobs.
`uv export --locked` fails if dependency metadata requires a lock update instead
of silently resolving new versions. Constraints intentionally omit hashes to
support the existing editable installation.

## Updating dependencies

Use a new shell without the CI constraint variables for lock updates.
After changing dependency declarations, run `uv lock` and commit both
`pyproject.toml` and `uv.lock`. This keeps compatible locked versions.
For a targeted update, run:

```bash
uv lock --upgrade-package PACKAGE
```

For routine maintenance, refresh the lock manually about once a month with
`uv lock --upgrade`, inspect the diff, and submit it through the normal CI matrix.
The existing seven-day PyPI release cooldown still applies to these updates.
Recreate local test environments after updating the lock.

When upgrading the pinned Hatch version, also synchronize the `ci` group's test
dependencies with `hatch.env.internal.test.get_default_config` for that release.
Keep the pinned uv version in that group and the workflow's setup-uv steps aligned.

The scheduled/manual `test-dev-deps` job deliberately stays outside these
constraints. It continues upgrading the scientific stack and PyTorch to their
development builds so upcoming compatibility problems remain visible.
