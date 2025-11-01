"""Telegram bot integration for AI Trading Beast."""
from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Optional

from loguru import logger
from telegram.ext import Application, CommandHandler

from core.engine import TradingEngine
from core.utils import AppConfig, load_config, load_env
from .handlers import BotHandlers


class TradingBot:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.engine = TradingEngine(self.config)
        self.handlers = BotHandlers(self.engine)
        env = load_env()
        token = env.get("TELEGRAM_BOT_TOKEN")
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN missing in environment")
        self.application = Application.builder().token(token).build()
        self._register_handlers()

    def _register_handlers(self) -> None:
        self.application.add_handler(CommandHandler("start", self.handlers.start))
        self.application.add_handler(CommandHandler("pause", self.handlers.pause))
        self.application.add_handler(CommandHandler("resume", self.handlers.resume))
        self.application.add_handler(CommandHandler("status", self.handlers.status))
        self.application.add_handler(CommandHandler("summary", self.handlers.summary))
        self.application.add_handler(CommandHandler("risk", self.handlers.risk))

    async def run(self) -> None:
        logger.info("AI Trading Beast starting Telegram bot")
        engine_task = asyncio.create_task(self.engine.run())
        await self.application.initialize()
        await self.application.start()
        try:
            await asyncio.gather(engine_task, self.application.updater.start_polling())
        finally:
            engine_task.cancel()
            with suppress(asyncio.CancelledError):
                await engine_task
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()


__all__ = ["TradingBot"]
