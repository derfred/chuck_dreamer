"""Click entry points for the object-localization pipeline."""
from .annotate_mat import annotate_mat_cmd
from .prompt_episodes import prompt_episodes_cmd

__all__ = [
  "annotate_mat_cmd",
  "prompt_episodes_cmd",
]
