import argparse
import base64
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Tuple, Optional

# Make GUI-Agent package importable despite hyphenated folder name
_THIS_DIR = os.path.dirname(__file__)
_GUI_AGENT_ROOT = os.path.join(_THIS_DIR, "GUI-Agent-main")
if _GUI_AGENT_ROOT not in sys.path:
    sys.path.insert(0, _GUI_AGENT_ROOT)

from agent.agent import FunctionCallAgent  # type: ignore
from browser_env.actions import ActionTypes  # type: ignore


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace('"', r'\"')


def _parse_coords_from_text(response_text: str) -> Optional[Tuple[float, float]]:
    if not response_text:
        return None
    nums = re.findall(r"-?\d+\.?\d*", response_text)
    if len(nums) >= 2:
        try:
            return float(nums[0]), float(nums[1])
        except Exception:
            return None
    return None


def _image_bytes_to_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")


class GUIAgentAdapter:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1500,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        grounding_base_url: Optional[str] = None,
        grounding_model_name: str = "ByteDance-Seed/UI-TARS-1.5-7B",
        grounding_api_key: str = "dummy-key",
    ) -> None:
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
        self._action_history: List[str] = []
        self._response_history: List[str] = []
        self._last_action_cmd: Optional[str] = None

        self.grounding_base_url = grounding_base_url
        self.grounding_model_name = grounding_model_name
        self.grounding_api_key = grounding_api_key
        if grounding_base_url:
            self.logger.info(f"Grounding enabled via {grounding_base_url}")

    def reset(self, logger: Optional[logging.Logger] = None) -> None:
        if logger is not None:
            self.logger = logger
        self._action_history.clear()
        self._response_history.clear()
        self._last_action_cmd = None

    def predict(self, instruction: str, obs: Dict[str, Any]) -> Tuple[str, List[Any]]:
        trajectory: List[Any] = []
        # Provide the latest screenshot to the agent as base64 so LLM can "see" the UI
        try:
            screenshot_bytes = obs.get("screenshot")
            if isinstance(screenshot_bytes, (bytes, bytearray)):
                trajectory.append({
                    "observation": {
                        "image": _image_bytes_to_b64(screenshot_bytes)
                    }
                })
        except Exception:
            # If screenshot is missing or malformed, proceed without trajectory image
            pass
        meta_data: Dict[str, Any] = {
            "action_history": self._action_history[-5:],
            "response_history": self._response_history[-5:],
        }
        action, _ = self.inner.next_action_custom(trajectory, instruction, meta_data)
        response_text, actions = self._convert_action(action, obs)
        pretty = response_text or str(action)
        self._response_history.append(pretty)
        if actions:
            self._action_history.append(str(actions[0]) if isinstance(actions, list) else str(actions))
        return pretty, actions

    def _convert_action(self, action: Dict[str, Any], obs: Dict[str, Any]) -> Tuple[str, List[Any]]:
        if not isinstance(action, dict):
            return "", ["WAIT"]

        a_type = action.get("action_type")

        def _is(t: ActionTypes) -> bool:
            try:
                return a_type == t or int(a_type) == int(t)
            except Exception:
                return a_type == t

        if _is(ActionTypes.WAIT):
            return "wait", ["WAIT"]

        if _is(ActionTypes.STOP):
            return action.get("answer", "DONE"), ["DONE"]

        if _is(ActionTypes.GOTO_URL):
            url = action.get("url", "")
            if not isinstance(url, str) or not url:
                return "goto_url:missing", ["WAIT"]
            cmd = (
                'pyautogui.hotkey("ctrl","l"); '
                'time.sleep(0.2); '
                f'pyautogui.typewrite("{_escape(url)}"); '
                'time.sleep(0.2); '
                'pyautogui.press("enter")'
            )
            return f"goto_url:{url}", [cmd, "WAIT"]

        if _is(ActionTypes.TYPE):
            text = action.get("text", "")
            if not isinstance(text, str):
                text = str(text)
            cmd = (
                f'pyautogui.typewrite("{_escape(text)}"); '
                'time.sleep(0.2); '
                'pyautogui.press("enter")'
            )
            return f"type:{text}", [cmd, "WAIT"]

        if _is(ActionTypes.KEY_PRESS):
            key = str(action.get("key_comb", "enter")).lower()
            key_map = {"enter": "enter", "delete": "delete", "space": "space"}
            press = key_map.get(key, key)
            cmd = f'pyautogui.press("{_escape(press)}")'
            actions = [cmd]
            if press in ("enter", "return"):
                actions.append("WAIT")
            return f"press:{press}", actions

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
            return f"scroll:{direction}", [cmd]

        if _is(ActionTypes.CLICK):
            desc = action.get("description", "")
            coords = self._ground_click(desc, obs)
            if coords:
                x, y = coords
                cmd = f"pyautogui.click({int(x)}, {int(y)})"
                return f"click:{desc}@({int(x)},{int(y)})", [cmd]
            self.logger.info(f"Grounding failed for CLICK: {desc}")
            return f"click(failed):{desc}", ["WAIT"]

        return "none", ["WAIT"]

    def _ground_click(self, description: str, obs: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        if not self.grounding_base_url:
            return None
        try:
            from urllib.request import Request, urlopen
            screenshot_bytes = obs.get("screenshot")
            if not isinstance(screenshot_bytes, (bytes, bytearray)):
                return None
            b64 = _image_bytes_to_b64(screenshot_bytes)
            system_msg = (
                "You are a grounding model, given the screenshot and the target element description, "
                "you need to identify the coordinates of the given element and return them in the "
                "format of click(point='<point>x1 y1</point>')."
            )
            user_items = [
                {"type": "text", "text": f"Target element description: {description}\nWhat's the coordinates of the target element in the screenshot? You should return as click(point='<point>x1 y1</point>')"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
            url = self.grounding_base_url.rstrip("/") + "/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.grounding_api_key:
                headers["Authorization"] = f"Bearer {self.grounding_api_key}"
            payload = json.dumps({
                "model": self.grounding_model_name,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_items},
                ],
                "temperature": 0.1,
                "max_tokens": 256,
            }).encode("utf-8")
            req = Request(url, data=payload, headers=headers, method="POST")
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            text = ""
            try:
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception:
                text = ""
            return _parse_coords_from_text(text)
        except Exception as e:
            self.logger.warning(f"Grounding click failed: {e}")
            return None

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

def _parse_coords_from_text(response_text: str) -> Optional[Tuple[float, float]]:
    """Parse two coordinates from grounding model response text."""
    if not response_text:
        return None
    # Strip everything except numbers, dot, comma, minus, whitespace
    nums = re.findall(r"-?\d+\.?\d*", response_text)
    if len(nums) >= 2:
        try:
            return float(nums[0]), float(nums[1])
        except Exception:
            return None
    return None


def _image_bytes_to_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")


 
