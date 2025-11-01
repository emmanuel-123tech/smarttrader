"""Telegram command handlers for AI Trading Beast."""
from __future__ import annotations


from telegram import Update
from telegram.ext import ContextTypes

from core.engine import TradingEngine
from core.utils import dumps_json


class BotHandlers:
    def __init__(self, engine: TradingEngine):
        self.engine = engine

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("🤖 AI Trading Beast online. Use /status for state.")

    async def pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.engine.pause()
        await update.message.reply_text("AI Trading Beast paused all activity.")

    async def resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.engine.resume()
        await update.message.reply_text("AI Trading Beast resumed trading.")

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        status = self.engine.status()
        await update.message.reply_text(f"AI Trading Beast status:\n{dumps_json(status)}")

    async def summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("AI Trading Beast summary not implemented yet.")

    async def risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        config = self.engine.config.risk
        await update.message.reply_text(f"AI Trading Beast risk settings:\n{dumps_json(config)}")


__all__ = ["BotHandlers"]
