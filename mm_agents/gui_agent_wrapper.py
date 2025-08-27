import argparse
from agent.agent import construct_agent

"""Wrapper class for GUI-Agent for compatibility with OSWorld evaluation"""

class GUIAgentWrapper:
    def __init__(self, args: argparse.Namespace):
        self.agent = construct_agent(args)
        self.trajectory = []
        self.meta_data = {}
        self.intent = ""
        self.args = args

    def predict(self, instruction, obs, **kwargs):
        # Store intent and meta_data for this step
        self.intent = instruction
        self.meta_data = kwargs.get("meta_data", {})
        # OSWorld expects a response and actions
        action = self.agent.next_action_custom(
            trajectory=self.trajectory,
            intent=self.intent,
            meta_data=self.meta_data,
            model=self.args.model,
            args=self.args
        )
        # For compatibility, wrap the action in a list and provide a dummy response
        response = str(action)
        actions = [action]
        # Update trajectory (if needed by your agent)
        self.trajectory.append({"instruction": instruction, "observation": obs, "action": action})
        return response, actions

    def reset(self, task_config=None, _logger=None):
        self.trajectory = []
        self.meta_data = {}
        self.intent = ""
        # Call your agent's reset method
        self.agent.reset(task_config if task_config is not None else "")