"""GLADE core: backend-agnostic optimization, plotting, format parsing and
translation, extracted from the duplicated legacy ``version_*.py`` scripts.

Sub-packages
------------
``core.format``     parse / merge / validate the V0.4 ``.dat`` configuration
``core.optimize``   (later) backend-agnostic Differential Evolution optimizer
``core.plot``       (later) unified result plotting
``core.translate``  (later) glafic <-> glade ``.dat`` translation
"""

__all__ = ["format"]
__version__ = "0.4.0-dev"
