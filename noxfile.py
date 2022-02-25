# pylint: skip-file

"""primary nox definition file."""

import nox

nox.options.sessions = ["lint", "tests"]


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("-r", "requirements.test.txt")
    session.install("-e", ".")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    session.install("-r", "requirements.test.txt")
    session.install("-e", ".")
    session.run("py.test", "src/python/pytorch-experimentation/tests", *session.posargs)
