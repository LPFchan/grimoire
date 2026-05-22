"""Plugin hooks — base class and manager."""

import logging

logger = logging.getLogger(__name__)


def env_flag(name, default):
    import os
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off", ""}


class Plugin:
    """Base plugin hook interface."""

    def __init__(self):
        self._enabled_override: bool | None = None

    def set_enabled(self, enabled: bool):
        self._enabled_override = enabled

    def _is_enabled(self) -> bool:
        return self._enabled_override if self._enabled_override is not None else self._default_enabled()

    def _default_enabled(self) -> bool:
        return False

    def info(self) -> dict:
        base = self._info()
        base["enabled"] = self._is_enabled()
        return base

    def _info(self) -> dict:
        return {"name": "base", "key": "", "description": ""}

    def before_request(self, payload, model_name, model_cfg):
        return payload

    def wrap_response_stream(self, stream, model_name, model_cfg):
        return stream

    async def before_backend_request(self, payload, model_name, model_cfg, backend_model_id, client, url, headers):
        return payload


class PluginManager:
    """Apply plugin hooks in a stable order."""

    def __init__(self, plugins):
        self.plugins = plugins

    def get_all_info(self) -> list[dict]:
        return [p.info() for p in self.plugins]

    def set_enabled(self, key: str, enabled: bool) -> dict | None:
        for p in self.plugins:
            if p.info().get("key") == key:
                p.set_enabled(enabled)
                return p.info()
        return None

    def before_request(self, payload, model_name, model_cfg):
        for plugin in self.plugins:
            payload = plugin.before_request(payload, model_name, model_cfg)
        return payload

    def wrap_response_stream(self, stream, model_name, model_cfg):
        for plugin in self.plugins:
            stream = plugin.wrap_response_stream(stream, model_name, model_cfg)
        return stream

    async def before_backend_request(self, payload, model_name, model_cfg, backend_model_id, client, url, headers):
        for plugin in self.plugins:
            payload = await plugin.before_backend_request(
                payload, model_name, model_cfg, backend_model_id, client, url, headers
            )
        return payload
