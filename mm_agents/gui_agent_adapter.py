import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

# Make GUI-Agent package importable despite hyphenated folder name
_THIS_DIR = os.path.dirname(__file__)
_GUI_AGENT_ROOT = os.path.join(_THIS_DIR, "GUI-Agent-main")
if _GUI_AGENT_ROOT not in sys.path:
    sys.path.insert(0, _GUI_AGENT_ROOT)

from agent.agent import FunctionCallAgent  # type: ignore
from browser_env.actions import ActionTypes  # type: ignore


class GUIAgentAdapter:
    """
    OSWorld adapter for the GUI-Agent FunctionCallAgent.

    Contract:
    - reset(logger?): initialize internal state.
    - predict(instruction, obs) -> (response_str, actions_list)
      where actions_list entries are strings or dicts accepted by DesktopEnv with action_space="pyautogui".
    """

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 1500, openai_api_key: str | None = None,
                 anthropic_api_key: str | None = None):
        # Build a lightweight args namespace expected by FunctionCallAgent
        args = argparse.Namespace(
            model=model,
            max_tokens=max_tokens,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            manual_action=False,
            collect_training_data=False,
        )
        self.inner = FunctionCallAgent(args)
        self.logger = logging.getLogger("desktopenv.agent.gui_adapter")
        # Track short histories for prompting
        self._action_history: List[str] = []
        self._response_history: List[str] = []

    # --- OSWorld Agent API ---
    def reset(self, logger: logging.Logger | None = None):
        if logger:
            self.logger = logger
        self._action_history.clear()
        self._response_history.clear()

    def predict(self, instruction: str, obs: Dict[str, Any]) -> Tuple[str, List[Any]]:
        # Prepare the minimal trajectory and meta_data expected by GUI-Agent
        trajectory: List[Any] = []  # We don't maintain a browser_env trajectory here
        meta_data: Dict[str, Any] = {
            "action_history": self._action_history[-5:],
            "response_history": self._response_history[-5:],
        }

        # Ask the inner agent for one action
        action, meta_out = self.inner.next_action_custom(trajectory, instruction, meta_data)

        # Convert GUI-Agent action to OSWorld pyautogui command(s)
        response_text, osworld_actions = self._convert_action(action, obs)

        # Book-keeping for next round
        pretty = response_text or str(action)
        self._response_history.append(pretty)
        if osworld_actions:
            # Only store a compact representation
            self._action_history.append(str(osworld_actions[0]) if isinstance(osworld_actions, list) else str(osworld_actions))

        return pretty, osworld_actions

    # --- Conversion helpers ---
    def _convert_action(self, action: Dict[str, Any], obs: Dict[str, Any]) -> Tuple[str, List[Any]]:
        if not isinstance(action, dict):
            return ("", ["WAIT"])  # guard

        a_type = action.get("action_type")

        # Normalize ActionType to name string for easier comparison
        def _is(t: ActionTypes) -> bool:
            try:
                return a_type == t or int(a_type) == int(t)
            except Exception:
                return a_type == t

        # WAIT
        if _is(ActionTypes.WAIT):
            return ("wait", ["WAIT"])  # env handles sleep via pause param

        # STOP -> DONE
        if _is(ActionTypes.STOP):
            return (action.get("answer", "DONE"), ["DONE"])  # mark task complete

        # GOTO_URL -> ctrl+l, type, enter
        if _is(ActionTypes.GOTO_URL):
            url = action.get("url", "")
            if not isinstance(url, str) or not url:
                return ("goto_url:missing", ["WAIT"])  # fallback
            cmd = f'pyautogui.hotkey("ctrl","l"); pyautogui.typewrite("{_escape(url)}"); pyautogui.press("enter")'
            return (f"goto_url:{url}", [cmd])

        # TYPE -> type text then press enter (assumes focus)
        if _is(ActionTypes.TYPE):
            text = action.get("text", "")
            if not isinstance(text, str):
                text = str(text)
            cmd = f'pyautogui.typewrite("{_escape(text)}"); pyautogui.press("enter")'
            return (f"type:{text}", [cmd])

        # KEY_PRESS -> map to pyautogui.press/hotkey
        if _is(ActionTypes.KEY_PRESS):
            key = str(action.get("key_comb", "enter")).lower()
            key_map = {"enter": "enter", "delete": "delete", "space": "space"}
            press = key_map.get(key, key)
            cmd = f'pyautogui.press("{_escape(press)}")'
            return (f"press:{press}", [cmd])

        # SCROLL -> pyautogui.scroll/hscroll
        if _is(ActionTypes.SCROLL):
            direction = str(action.get("direction", "down")).lower()
            if direction in ("down", "up"):
                amount = -800 if direction == "down" else 800
                cmd = f"pyautogui.scroll({amount})"
            elif direction in ("left", "right"):
                amount = -600 if direction == "left" else 600
                cmd = f"pyautogui.hscroll({amount})"
            else:
                cmd = "pyautogui.scroll(-600)"
            return (f"scroll:{direction}", [cmd])

        # CLICK -> not grounded; skip safely with WAIT for now
        if _is(ActionTypes.CLICK):
            desc = action.get("description", "")
            self.logger.info(f"Skipping CLICK without grounding: {desc}")
            return (f"click(skipped):{desc}", ["WAIT"])  # noop placeholder

        # Fallback
        return ("none", ["WAIT"])  # be conservative


def _escape(text: str) -> str:
    """Escape quotes/backslashes for safe inclusion in pyautogui string literals."""
    return text.replace("\\", r"\\").replace('"', r'\"')
