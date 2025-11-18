"""
Visualization modules for perception system.

Provides:
- Overlay rendering for detection results
- Bird's eye view generation (coming in Phase 3)
- Minimap visualization (coming in Phase 3)
- Real-time plotting widgets (coming in Phase 3)
"""

from .overlay_renderer import OverlayRenderer, Colors

__all__ = [
    'OverlayRenderer',
    'Colors',
]
