"""Plugin hooks for Grimoire request and response transforms."""

import json

from grimoire.plugins.base import Plugin, PluginManager
from grimoire.plugins.tool_arg_sanitize import ToolArgSanitizePlugin
from grimoire.plugins.buun_template import BuunTemplatePlugin
from grimoire.plugins.structured_cot import QwenStructuredCotPlugin
from grimoire.plugins.pflash_awareness import PflashAwarenessPlugin
from grimoire.plugins.tool_plan import StructuredToolPlanPlugin

plugin_manager = PluginManager([
    # Runs first: clean malformed tool-call args before any downstream transform
    # or the backend's func_args_not_string sees them.
    ToolArgSanitizePlugin(),
    BuunTemplatePlugin(),
    QwenStructuredCotPlugin(),
    PflashAwarenessPlugin(),
    StructuredToolPlanPlugin(),
])


def restore_plugin_states(user_hash: str, store):
    """Restore persisted plugin toggle states from the settings store."""
    try:
        all_settings = store.get_all(user_hash)
        for key, raw in all_settings.items():
            if key.startswith("plugin."):
                plugin_key = key[len("plugin."):]
                try:
                    enabled = json.loads(raw)
                    plugin_manager.set_enabled(plugin_key, enabled)
                except (json.JSONDecodeError, ValueError):
                    pass
    except Exception:
        pass
