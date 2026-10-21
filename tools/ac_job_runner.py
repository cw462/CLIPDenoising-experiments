"""Persist a subprocess exit record for the resumable A/C experiment queue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exit-file', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A subprocess command is required')
    started = time.time()
    child = subprocess.Popen(command)
    code = child.wait()
    path = Path(args.exit_file)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'returncode': code, 'command': command,
                                    'started_at_unix': started, 'ended_at_unix': time.time(),
                                    'child_pid': child.pid, 'wrapper_pid': os.getpid()}), encoding='utf-8')
    os.replace(temporary, path)
    return code


if __name__ == '__main__':
    sys.exit(main())
