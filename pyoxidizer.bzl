load("pyoxidizer", "python_distribution", "python_module", "package_data")

def make_python_package():
    return python_distribution(
        name = "server",
        packages = [
            python_module("modules.main"),
            python_module("configs.registries.optimizerreg"),
            python_module("configs.registries.lossreg"),
            python_module("configs.registries.metricreg"),
            python_module("configs.registries.layerreg"),
            python_module("configs.registries.modelreg"),
            python_module("fastapi"),
            python_module("fastapi.middleware.cors"),
            python_module("fastapi.middleware.httpsredirect"),
            python_module("fastapi.middleware.trustedhost"),
            python_module("uvicorn"),
            python_module("uvicorn.logging"),
            python_module("uvicorn.loops.auto"),
            python_module("uvicorn.protocols.http.auto"),
            python_module("uvicorn.protocols.websockets.auto"),
            python_module("uvicorn.lifespan.on"),
            python_module("pydantic"),
            python_module("encodings"),
        ],
        data_files = [
            package_data("templates", "templates/*"),
            package_data("modules", "modules/*"),
            package_data(".", "pyproject.toml"),
        ],
        build_mode = "onefile",
    )
