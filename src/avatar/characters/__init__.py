"""Character style definitions."""
from .stick_figure import StickFigureStyle
from .silhouette import SilhouetteStyle
from .neon import NeonStyle
from .cartoon import CartoonStyle
from .ghost import GhostStyle
from .puppet import PuppetStyle, CHARACTER_PRESETS
from .mesh3d import Mesh3DStyle, list_available_models
from .deform2d import Deform2DStyle, DEFORM_PRESETS
from .hqchar_v2 import HQCharacterStyle, HQCHAR_PRESETS

STYLE_REGISTRY = {
    "stick_figure": StickFigureStyle,
    "silhouette": SilhouetteStyle,
    "neon": NeonStyle,
    "cartoon": CartoonStyle,
    "ghost": GhostStyle,
    "puppet": PuppetStyle,
    "mesh3d": Mesh3DStyle,
    "deform2d": Deform2DStyle,
    "hqchar": HQCharacterStyle,
}

__all__ = [
    "StickFigureStyle", "SilhouetteStyle", "NeonStyle", "CartoonStyle",
    "GhostStyle", "PuppetStyle", "Mesh3DStyle",
    "STYLE_REGISTRY", "CHARACTER_PRESETS", "list_available_models",
]
