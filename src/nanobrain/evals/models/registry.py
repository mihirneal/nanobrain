"""Model registry for the evals.

Each model is a module in this package with a constructor registered by name:

```
@register_model
def new_model(*, ckpt_path, random_init=False, **kwargs):
    ...
    return transform, model
```
"""

import importlib
import logging
import pkgutil

import nanobrain.evals.models
from nanobrain.evals.models.base import ModelFn, ModelTransformPair

_logger = logging.getLogger(__package__)

_MODEL_REGISTRY: dict[str, ModelFn] = {}


def register_model(name_or_func: str | ModelFn | None = None):
    def _decorator(func: ModelFn):
        name = name_or_func if isinstance(name_or_func, str) else func.__name__
        if name in _MODEL_REGISTRY:
            _logger.warning(f"Model {name} already registered; overwriting.")
        _MODEL_REGISTRY[name] = func
        return func

    if callable(name_or_func):
        return _decorator(name_or_func)
    return _decorator


def create_model(name: str, **kwargs) -> ModelTransformPair:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model {name} not registered, available: {list_models()}")
    transform, model = _MODEL_REGISTRY[name](**kwargs)
    return transform, model


def list_models() -> list[str]:
    return list(_MODEL_REGISTRY)


def import_model_plugins():
    """Finds and imports all plugins registering new models."""
    # https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-namespace-packages
    plugins = {}
    for finder, name, ispkg in pkgutil.iter_modules(nanobrain.evals.models.__path__):
        if not (name in {"base", "registry"} or name.startswith("test_")):
            try:
                plugins[name] = importlib.import_module(f"nanobrain.evals.models.{name}")
            except Exception as exc:
                _logger.warning(f"Import model plugin {name} failed: {exc}")
    return plugins


# import all discovered plugins to register
_MODEL_PLUGINS = import_model_plugins()
