### PR recommendations

To allow your work to be integrated as seamlessly as possible, we advise you to:

- **Create a new branch** for your PR. This will allow you to keep your changes separate from the `main` branch and
  facilitate the review process.

- **Keep your PR small**. If you want to contribute a new feature, consider splitting it into multiple PRs. This will
  allow us to review your work more easily and provide you with feedback faster.

- Verify your PR is **up-to-date** with `safevideo/autollm` `main` branch. If your PR is behind you can update
  your code by clicking the 'Update branch' button or by running `git pull` and `git merge main` locally.

### Docstrings

Not all functions or classes require docstrings but when they do, we
follow [google-style docstrings format](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
Here is an example:

```python
"""
    What the function does. Performs NMS on given detection predictions.

    Args:
        arg1: The description of the 1st argument
        arg2: The description of the 2nd argument

    Returns:
        What the function returns. Empty if nothing is returned.

    Raises:
        Exception Class: When and why this exception can be raised by the function.
"""
```

### Code style

We use pre-commit hooks to ensure that all code is formatted according to
[flake8](https://flake8.pycqa.org/en/latest/) and
[isort](https://pycqa.github.io/isort/). To install the pre-commit hooks, run:

```bash
pre-commit install
```
