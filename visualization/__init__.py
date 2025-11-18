"""
Visualization modules for perception system.

Provides:
- Overlay rendering for detection results
- Bird's eye view generation
- Minimap visualization
- Real-time plotting widgets (coming in Phase 4)
"""

from .overlay_renderer import OverlayRenderer, Colors
from .bev_generator import BEVGenerator, BEVConfig
from .minimap import MinimapWidget

__all__ = [
    'OverlayRenderer',
    'Colors',
    'BEVGenerator',
    'BEVConfig',
    'MinimapWidget',
]
