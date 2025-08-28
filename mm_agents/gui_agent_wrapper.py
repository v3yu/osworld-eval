from agent.agent import FunctionCallAgent

class GUIAgentWrapper:
    def __init__(self, args):
        self.agent = FunctionCallAgent(args)
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