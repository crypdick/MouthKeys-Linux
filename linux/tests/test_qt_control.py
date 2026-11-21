import subprocess
import sys

import pytest

from mouthkeys import qt_control


@pytest.mark.parametrize("fail", [False, True])
def test_qt_request_forwards_args_and_reports_outcome_without_logging(monkeypatch, capsys, fail):
    calls = []

    def request(action, **data):
        calls.append((action, data))
        if fail:
            raise RuntimeError("Daemon unavailable")
        return {"phase": "idle"}

    monkeypatch.setattr(qt_control, "request", request)
    job = qt_control.Request("copy", {"id": "example"})
    responses = []
    job.done.connect(responses.append)
    job.run()
    assert calls == [("copy", {"id": "example"})]
    expected = {"error": "Daemon unavailable"} if fail else {"result": {"phase": "idle"}}
    assert responses == [expected]
    output = capsys.readouterr()
    assert not output.out
    assert not output.err


def test_passive_interfaces_do_not_import_settings(tmp_path):
    script = (
        "import sys; from mouthkeys import overlay, tray; assert 'mouthkeys.gui' not in sys.modules"
    )
    subprocess.run(  # noqa: S603
        [sys.executable, "-P", "-c", script],
        cwd=tmp_path,
        timeout=5,
        capture_output=True,
        check=True,
    )
