"""Use the Pydantic generation that owns the installed Prefect Block model."""

from prefect.blocks.core import Block
from pydantic import BaseModel

if issubclass(Block, BaseModel):
    from pydantic import Field, PrivateAttr
else:
    from pydantic.v1 import Field, PrivateAttr

__all__ = ["Field", "PrivateAttr"]
