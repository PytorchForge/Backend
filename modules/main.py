import os
if 'PYTHONHOME' in os.environ:
   del os.environ['PYTHONHOME']

import argparse
import uvicorn
import sys


from pathlib import Path
if getattr(sys, 'frozen', False):
    sys.path.insert(0, str(Path(sys._MEIPASS) / "modules")) 
    sys.path.insert(0, str(Path(sys._MEIPASS) / "templates"))
    sys.path.insert(0, str(Path(sys._MEIPASS) / "static"))
else:
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent / "templates"))
print(f"Base directory in main is: {Path(__file__).parent}", file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(prog="run")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--reload", action="store_true")
    p.add_argument("--log-level", default="info")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--lifespan", choices=("on", "off", "auto"), default="on")
    p.add_argument("--loop", default="auto")
    return p.parse_known_args()[0]

def main(): 
    args = parse_args()
    is_frozen = getattr(sys, "frozen", False)

    if args.reload and args.workers != 1:
        raise SystemExit("cannot use --reload with multiple workers")
    

    uvicorn.run(
        app="server:app",
        host=args.host,
        port=args.port,
        reload=(args.reload and not is_frozen),
        reload_dirs=["modules", "templates"],
        log_level=args.log_level,
        workers=args.workers,
        lifespan=args.lifespan,
        loop=args.loop,
    )

if __name__ == "__main__":
    main()
