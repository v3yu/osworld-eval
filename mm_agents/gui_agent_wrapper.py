from agent.agent import FunctionCallAgent
from tools.gui_tools import ClickTool, TypeTool, ScrollTool, WaitTool, StopTool, PressKeyTool, PageGotoTool
from tools.analysis_tools import MapSearchTool, ContentAnalyzerTool
from tools.web_search_tools import WebSearchTool

class PatchedFunctionCallAgent(FunctionCallAgent):
    def _define_functions(self):
        # Return tool instances with correct names
        click_tool = ClickTool(); click_tool.name = "click"
        type_tool = TypeTool(); type_tool.name = "type"
        press_key_tool = PressKeyTool(); press_key_tool.name = "press_key"
        scroll_tool = ScrollTool(); scroll_tool.name = "scroll"
        wait_tool = WaitTool(); wait_tool.name = "wait"
        stop_tool = StopTool(); stop_tool.name = "stop"
        map_search_tool = MapSearchTool(); map_search_tool.name = "map_search"
        content_analyzer_tool = ContentAnalyzerTool(); content_analyzer_tool.name = "content_analyzer"
        page_goto_tool = PageGotoTool(); page_goto_tool.name = "goto_url"
        # web_search_tool = WebSearchTool(); web_search_tool.name = "google_web_search" # Uncomment if needed

        return [
            click_tool,
            type_tool,
            press_key_tool,
            scroll_tool,
            wait_tool,
            stop_tool,
            map_search_tool,
            content_analyzer_tool,
            page_goto_tool,
            # web_search_tool, # Uncomment if needed
        ]

class GUIAgentWrapper:
    def __init__(self, args):
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