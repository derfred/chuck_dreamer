"""Real-hardware recording: scene registration + episode capture via lerobot.

The CLI entry point is ``chuck-dreamer record`` (see :mod:`main`). This package
orchestrates two phases:

  * **Phase A — scene registration** (:mod:`scene_registration`): per-camera
    intrinsics from a checkerboard, then a short video clip per named marker
    location for noise characterisation.
  * **Phase B — episode recording**: delegates to upstream
    :func:`lerobot.scripts.lerobot_record.record` with a config built from
    ``cfg.recording`` by :mod:`config_builders`.

Output goes through a :class:`storage.SessionStorage` backend selectable from
the CLI (``--storage lerobot|episodes``).
"""

from .recorder import run

__all__ = ["run"]
