# Pre-commit git hooks

Git hooks to integrate with [pre-commit](http://pre-commit.com).


<!--TOC-->

- [Configure pre-commit](#configure-pre-commit)
- [Two ways to invoke pre-commit](#two-ways-to-invoke-pre-commit)
- [Available hooks](#available-hooks)
  - [`no-commit`](#no-commit)
<!-- - [Testing](#testing) -->
- [License](#license)
- [Credits](#credits)

<!--TOC-->


## Configure pre-commit

Add to `.pre-commit-config.yaml` in your git repo:

    - repo: https://github.com/abaisero/pre-commit-hooks
      rev: main  # or specific git tag
      hooks:
        - id: no-commit


## Two ways to invoke pre-commit

If you want to invoke the checks as a git pre-commit hook, run:

    pre-commit install

If you want to run the checks on-demand (outside of git hooks), run:

    pre-commit run --all-files --verbose

<!-- The [test harness](TESTING.md) of this git repo uses the second approach -->
<!-- to run the checks on-demand. -->


## Available hooks

### `no-commit`

**What it does**

Checks that text files do **not** contain a pre-specified strings (which
defaults to "NOCOMMIT").  This is useful to tag parts of the code which needs
to be revisited before being commited.

**Default**

Filter on files that are `text`, using .

  types: [text]
  args: [--tag=NOCOMMIT]

**Custom configuration (overrides)**

Use the [built-in
overrides](https://pre-commit.com/#pre-commit-configyaml---hooks) from the
pre-commit framework.


<!-- ## Testing -->

<!-- Please see [TESTING.md](TESTING.md). -->


## License

The code in this repo is licensed under the [MIT License](LICENSE).


## Credits

The structure of this reposiroty is based on
[jumanjihouse/pre-commit-hooks](https://https://github.com/humanjihouse/pre-commit-hooks).
