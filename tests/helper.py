import json
import os
import shlex
import subprocess

_PATH_TESTS_DIR = os.path.dirname(__file__)
TEST_SCRIPT = os.path.join(_PATH_TESTS_DIR, "train.py")


def _run_galvatron(trainer_options, strategy_options):
    """This function executes `tests/train.py::run_test_from_config` in a new subprocess The behavior influences the
    pytest coverage report, because coverage cannot track what subprocess did.
    """
    cmdline = [
        "python3",
        TEST_SCRIPT,
        "--trainer-options",
        shlex.quote(json.dumps(trainer_options)),
        "--strategy-options",
        shlex.quote(json.dumps(strategy_options)),
    ]
    exit_code = subprocess.call(" ".join(cmdline), shell=True, env=os.environ.copy())
    assert exit_code == 0
