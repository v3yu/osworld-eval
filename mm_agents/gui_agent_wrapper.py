import argparse
from agent.agent import FunctionCallAgent
from tools.gui_tools import ClickTool, TypeTool, ScrollTool, WaitTool, StopTool, PressKeyTool, PageGotoTool
from tools.analysis_tools import MapSearchTool, ContentAnalyzerTool
from tools.web_search_tools import WebSearchTool

"""Wrapper class for GUI-Agent for compatibility with OSWorld evaluation"""

class PatchedFunctionCallAgent(FunctionCallAgent):
    def _define_functions(self):
        # Return tool instances, not strings
        return [
            ClickTool(),
            TypeTool(),
            ScrollTool(),
            WaitTool(),
            StopTool(),
            PressKeyTool(),
            MapSearchTool(),
            ContentAnalyzerTool(),
            PageGotoTool(),
            # WebSearchTool(), # Uncomment if needed
        ]

class GUIAgentWrapper:
    def __init__(self, args: argparse.Namespace):
        self.agent = PatchedFunctionCallAgent(args)
        self.trajectory = []
        self.meta_data = {}
        self.intent = ""
        self.args = args

    def predict(self, instruction, obs, **kwargs):
        self.intent = instruction
        self.meta_data = kwargs.get("meta_data", {})
        action = self.agent.next_action_custom(
            trajectory=self.trajectory,
            intent=self.intent,
            meta_data=self.meta_data,
            model=self.args.model,
            args=self.args
        )
        response = str(action)
        actions = [action]
        self.trajectory.append({"instruction": instruction, "observation": obs, "action": action})
        return response, actions

    def reset(self, task_config=None, _logger=None):
        self.trajectory = []
        self.meta_data = {}
        self.intent = ""
        self.agent.reset(task_config if task_config is not None else "")