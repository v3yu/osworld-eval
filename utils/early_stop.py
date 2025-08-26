"""Early stop functionality for the GUI Agent"""
from beartype import beartype
from browser_env import Trajectory, ActionTypes, Action
from browser_env.actions import is_equivalent


@beartype
def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to early stop"""

    last_k_actions: list[Action]
    action_seq: list[Action]
    action_seq = [item for item in trajectory if item.get('action_type', '')!='']
    
    # reach the max step
    # non_scroll_actions = [action for action in action_seq if action["action_type"] not in [ActionTypes.SCROLL, 'scroll']]
    # num_steps = len(non_scroll_actions)
    num_steps = len(action_seq)
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = action_seq[-k:]  # type: ignore[assignment]
    # print('trajectory: ', trajectory)
    # print('last_k_actions: ', last_k_actions)
    for idx, action in enumerate(last_k_actions):
        if action.get('action_type', '') == '':
            print(f"Action {idx} in last_k_actions is empty: {action}")
    if len(last_k_actions) >= k:
        if all(
            [
                action.get('action_type', '') == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # # Case: same action for k times
    # k = thresholds["repeating_action"]
    # last_k_actions = action_seq[-k:]  # type: ignore[assignment]

    # if len(action_seq) == 0:
    #     return False, ""

    # last_action: Action = action_seq[-1]

    # # if last_action["action_type"] not in [ActionTypes.TYPE, 'type', ActionTypes.SCROLL, 'scroll']:
    # if len(last_k_actions) >= k:
    #     if all(
    #         [
    #             is_equivalent(action, last_action)
    #             for action in last_k_actions
    #         ]
    #     ):
    #         return True, f"Same action {last_action['action_type']} for {k} times"

    return False, "" 