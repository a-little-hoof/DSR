"""Encoder registry for stage 1 models."""

from typing import Callable, Dict, Optional, Type, Union
import importlib.util
from pathlib import Path
import sys

ARCHS: Dict[str, Type] = {}
__all__ = ["ARCHS", "register_encoder"]


def _add_to_registry(name: str, cls: Type) -> Type:
    if name in ARCHS and ARCHS[name] is not cls:
        raise ValueError(f"Encoder '{name}' is already registered.")
    ARCHS[name] = cls
    return cls


def register_encoder(cls: Optional[Type] = None, *, name: Optional[str] = None) -> Union[Callable[[Type], Type], Type]:
    """Register an encoder class in ``ARCHS``.

    Can be used either as ``@register_encoder()`` (optionally passing ``name``) or
    via ``register_encoder(MyClass)`` after the class definition.
    """

    def decorator(inner_cls: Type) -> Type:
        encoder_name = name or inner_cls.__name__
        return _add_to_registry(encoder_name, inner_cls)

    if cls is None:
        return decorator

    return decorator(cls)


# Import modules that perform registration on import.
from . import dinov2  
from . import siglip2
from . import mae
from . import dinov3
from . import dinov2_no_reg
from . import siglip2_tt_reg

_tt_reg_encoder_name = "stage1.encoders.siglip2_tt_reg_encoder"
if _tt_reg_encoder_name not in sys.modules:
    _tt_reg_encoder_path = Path(__file__).resolve().parent / "siglip2_tt_reg.py"
    _spec = importlib.util.spec_from_file_location(_tt_reg_encoder_name, _tt_reg_encoder_path)
    if _spec and _spec.loader:
        _module = importlib.util.module_from_spec(_spec)
        sys.modules[_tt_reg_encoder_name] = _module
        _spec.loader.exec_module(_module)
