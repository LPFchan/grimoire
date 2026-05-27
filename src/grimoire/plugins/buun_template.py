"""Buun Qwen3.6 chat template plugin — toggles the hardened Jinja template for Qwen-family models."""

import asyncio
import logging
import os

from grimoire.plugins.base import Plugin

logger = logging.getLogger(__name__)

BUUN_TEMPLATE_PATH = "/templates/buun-Qwen3.6-chat_template.jinja"
QWEN_FAMILIES = {"qwen"}


class BuunTemplatePlugin(Plugin):
    """Toggle the buun Qwen3.6 chat template on/off for Qwen-family models.

    When enabled, sets ``chat-template-file`` on all Qwen-family models in the
    registry and restarts any active Qwen models so they pick up the new template.
    When disabled, clears ``chat-template-file`` and restarts.

    The seed ``etc/models.json`` should have ``chat-template-file`` set for all
    Qwen models so that the default (enabled) state is correct on first startup.
    """

    def _default_enabled(self) -> bool:
        return True

    def _info(self) -> dict:
        return {
            "name": "Buun Qwen3.6 Chat Template",
            "key": "BUUN_TEMPLATE",
            "description": "Uses the hardened buun Qwen3.6 chat template (25 fixes over official) for all Qwen-family models. Toggling restarts active Qwen models.",
        }

    def set_enabled(self, enabled: bool):
        previous = self._enabled_override
        self._enabled_override = enabled
        if previous != enabled:
            self._schedule_apply(enabled)

    def _schedule_apply(self, enabled: bool):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._apply(enabled))
        except RuntimeError:
            logger.warning("BUUN_TEMPLATE: No running event loop; cannot restart models")

    async def _apply(self, enabled: bool):
        from grimoire.registry import registry
        from grimoire.entrypoint import manager

        if enabled and not os.path.exists(BUUN_TEMPLATE_PATH):
            logger.error("BUUN_TEMPLATE: Template file not found at %s", BUUN_TEMPLATE_PATH)
            return

        needs_restart: list[str] = []
        for name in registry.list_all():
            cfg = registry.get(name)
            if not cfg or cfg.get("family") not in QWEN_FAMILIES:
                continue
            current = cfg.get("chat-template-file")
            if enabled and current != BUUN_TEMPLATE_PATH:
                try:
                    registry.update(name, {"chat-template-file": BUUN_TEMPLATE_PATH})
                    needs_restart.append(name)
                    logger.info("BUUN_TEMPLATE: Enabled template for %s", name)
                except Exception:
                    logger.exception("BUUN_TEMPLATE: Failed to update registry for %s", name)
            elif not enabled and current is not None:
                try:
                    registry.update(name, {"chat-template-file": None})
                    needs_restart.append(name)
                    logger.info("BUUN_TEMPLATE: Disabled template for %s", name)
                except Exception:
                    logger.exception("BUUN_TEMPLATE: Failed to update registry for %s", name)

        for name in needs_restart:
            if manager.get_active(name):
                try:
                    await manager.start_model(name)
                    logger.info("BUUN_TEMPLATE: Restarted %s (template %s)", name,
                                "enabled" if enabled else "disabled")
                except Exception:
                    logger.exception("BUUN_TEMPLATE: Failed to restart %s", name)
