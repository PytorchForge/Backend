import argparse
import uvicorn

def parse_args():
    p = argparse.ArgumentParser(prog="run")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.add_argument("--log-level", default="info")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--lifespan", choices=("on","off","auto"), default="on")
    p.add_argument("--loop", default="auto")
    return p.parse_args()

def main():
    args = parse_args()
    if args.reload and args.workers != 1:
        raise SystemExit("cannot use --reload with multiple workers")
    uv_cfg = dict(
        app="modules.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        workers=args.workers,
        lifespan=args.lifespan,
        loop=args.loop,
    )
    uvicorn.run(**uv_cfg)

if __name__ == "__main__":
    main()


# placeholder for binary version