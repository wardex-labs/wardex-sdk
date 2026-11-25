"""Whether an adapter attaches to the installed framework and says what it should.

Run inside a point's environment, with the real table of tested versions: on
a release inside the table the adapter installs, reports `measured=True` and
says nothing; on any other release it still installs, reports
`measured=False` and says so once. This is the "attach and flag" promise,
proven for the exact version the point installed. Prints one JSON object.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys


def main(adapter: str, modules: list[str]) -> int:
    import wardex_sdk as wardex
    from wardex_sdk import AdapterName
    from wardex_sdk._adapters._measured import MEASURED, installed_version, is_measured

    said: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                said.append(record.getMessage())

    logging.getLogger("wardex_sdk").addHandler(Collect())
    wardex.init(transport=wardex.NoOpTransport())
    try:
        for module in modules:
            importlib.import_module(module)
        status = next(s for s in wardex.diagnostics().adapters if s.name == adapter)
    finally:
        wardex.close()

    entries = MEASURED[AdapterName(adapter)]
    versions = {entry.distribution: installed_version(entry.distribution) for entry in entries}
    in_table = all(
        versions[entry.distribution] is not None
        and is_measured(entry, versions[entry.distribution])
        for entry in entries
    )
    about = [line for line in said if any(entry.distribution in line for entry in entries)]
    print(
        json.dumps(
            {
                "state": status.state.value,
                "version": status.version,
                "measured": status.measured,
                "in_table": in_table,
                "lines_said": len(about),
                "detail": status.detail[:300],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2].split(",")))
