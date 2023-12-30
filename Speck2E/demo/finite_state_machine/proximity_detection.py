from finite_state_machine.finite_state_machine import (
    Ret,
    State,
    StateMachine,
    state_machine_handle_event,
    state_machine_init,
    Transition,
)

from enum import IntEnum
from typing import Dict, List


SILENT_NUM_THRESH = 3


class Label(IntEnum):
    silent = -1
    m3 = 0
    m2 = 1
    m1 = 2
    left = 3
    right = 4


class Output(IntEnum):
    silent = 0
    near_3m = 1
    near_2m = 2
    near_1m = 3
    away_2m = 4
    away_3m = 5
    left = 6
    right = 7


def state_idle_data_init(state: State):
    state.data: List[int] = [Label.silent]
    state.data_count: Dict[int, int] = {Label.silent: 1, Label.m3: 0, Label.m2: 0, Label.m1: 0, Label.left: 0, Label.right: 0}


def state_near_3m_data_init(state: State):
    state.data: List[int] = [Label.m3]
    state.data_count: Dict[int, int] = {Label.silent: 0, Label.m3: 1, Label.m2: 0, Label.m1: 0, Label.left: 0, Label.right: 0}


def state_near_2m_data_init(state: State):
    state.data: List[int] = [Label.m2]
    state.data_count: Dict[int, int] = {Label.silent: 0, Label.m3: 0, Label.m2: 1, Label.m1: 0, Label.left: 0, Label.right: 0}


def state_near_1m_data_init(state: State):
    state.data: List[int] = [Label.m1]
    state.data_count: Dict[int, int] = {Label.silent: 0, Label.m3: 0, Label.m2: 0, Label.m1: 1, Label.left: 0, Label.right: 0}


def state_away_2m_data_init(state: State):
    state.data: List[int] = [Label.m2]
    state.data_count: Dict[int, int] = {Label.silent: 0, Label.m3: 0, Label.m2: 1, Label.m1: 0, Label.left: 0, Label.right: 0}


def state_away_3m_data_init(state: State):
    state.data: List[int] = [Label.m3]
    state.data_count: Dict[int, int] = {Label.silent: 0, Label.m3: 1, Label.m2: 0, Label.m1: 0, Label.left: 0, Label.right: 0}


def state_left_data_init(state: State):
    state.data: List[int] = [Label.left]
    state.data_count: Dict[int, int] = {Label.silent: 0, Label.m3: 0, Label.m2: 0, Label.m1: 0, Label.left: 1, Label.right: 0}


def state_right_data_init(state: State):
    state.data: List[int] = [Label.right]
    state.data_count: Dict[int, int] = {Label.silent: 0, Label.m3: 0, Label.m2: 0, Label.m1: 0, Label.left: 0, Label.right: 1}


def state_idle_action(state: State, feature: int) -> int:
    '''
    The current state is idle, data = [silent]. And the next state may be 
    1) left, silent -> left
    2) right, silent -> right
    3) near 3m, silent -> m3
    4) away 3m, silent -> m2 -> m3
    5) near 1m, silent -> m2 -> m1
    6) away 2m, silent -> m1 -> m2
    7) idle
    '''
    if not (feature in state.data):
        state.data.append(feature)

    state.data_count[feature] += 1

    if state.data_count[Label.left] > 1:  # silent -> left
        return Output.left
    elif state.data_count[Label.right] > 1:  # silent -> right
        return Output.right
    elif state.data_count[Label.m3] > 2:  # silent -> m3
        return Output.near_3m
    elif state.data_count[Label.m1] > 2 and state.data_count[Label.m2] > 0 and state.data.index(Label.m2) < state.data.index(Label.m1):  # silent -> m2 -> m1
        return Output.near_1m
    elif state.data_count[Label.m3] > 2 and state.data_count[Label.m2] > 0 and state.data.index(Label.m2) < state.data.index(Label.m3):  # silent -> m2 -> m3
        return Output.away_3m
    elif state.data_count[Label.m2] > 2 and state.data_count[Label.m1] > 0 and state.data.index(Label.m1) < state.data.index(Label.m2):  # silent -> m1 -> m2
        return Output.away_2m
    else:
        pass

    return Ret.EMPTY


def state_near_3m_action(state: State, feature: int) -> int:
    '''
    The current state is near 3m, data = [m3]. And the next state may be
    1) near 2m, m3 -> m2
    2) idle, m3 -> silent or other
    '''
    if not (feature in state.data):
        state.data.append(feature)

    state.data_count[feature] += 1

    if state.data_count[Label.silent] > SILENT_NUM_THRESH:  # m3 -> silent
        return Output.silent
    elif state.data_count[Label.m2] > 2 and state.data_count[Label.m3] > 0 and state.data.index(Label.m3) < state.data.index(Label.m2):  # m3 -> m2
        return Output.near_2m
    else:
        pass

    return Ret.EMPTY


def state_near_2m_action(state: State, feature: int) -> int:
    '''
    The current state is near 2m, data = [m2]. And the next state may be
    1) near 1m, m2 -> m1,
    2) away 3m, m2 -> m3,
    3) idle, m2 -> silent or other.
    '''
    if not (feature in state.data):
        state.data.append(feature)

    state.data_count[feature] += 1

    if state.data_count[Label.silent] > SILENT_NUM_THRESH:  # m2 -> silent
        return Output.silent
    elif state.data_count[Label.m1] > 2 and state.data_count[Label.m2] > 0 and state.data.index(Label.m2) < state.data.index(Label.m1):  # m2 -> m1
        return Output.near_1m
    elif state.data_count[Label.m3] > 2 and state.data_count[Label.m2] > 0 and state.data.index(Label.m2) < state.data.index(Label.m3):  # m2 -> m3
        return Output.away_3m
    else:
        pass

    return Ret.EMPTY


def state_near_1m_action(state: State, feature: int) -> int:
    '''
    The current state is near 1m, data = [m1]. And the next state may be
    1) away 2m, m1 -> m2
    2) idle, m1 -> silent
    '''
    if not (feature in state.data):
        state.data.append(feature)

    state.data_count[feature] += 1

    if state.data_count[Label.silent] > SILENT_NUM_THRESH:  # m1 -> silent
        return Output.silent
    elif state.data_count[Label.m2] > 2 and state.data_count[Label.m1] > 0 and state.data.index(Label.m1) < state.data.index(Label.m2):  # m1 -> m2
        return Output.away_2m
    else:
        pass

    return Ret.EMPTY


def state_away_2m_action(state: State, feature: int) -> int:
    '''
    The current state is away 2m, data = [m2]. And the next state may be
    1) near 1m, m2 -> m1,
    2) away 3m, m2 -> m3,
    3) idle, m2 -> silent or other.
    '''
    if not (feature in state.data):
        state.data.append(feature)

    state.data_count[feature] += 1

    if state.data_count[Label.silent] > SILENT_NUM_THRESH:    # m2 -> silent
        return Output.silent
    elif state.data_count[Label.m1] > 2 and state.data_count[Label.m2] > 0 and state.data.index(Label.m2) < state.data.index(Label.m1):  # m2 -> m1
        return Output.near_1m
    elif state.data_count[Label.m3] > 2 and state.data_count[Label.m2] > 0 and state.data.index(Label.m2) < state.data.index(Label.m3):  # m2 -> m3
        return Output.away_3m
    else:
        pass

    return Ret.EMPTY


def state_away_3m_action(state: State, feature: int) -> int:
    '''
    The current state is away 3m, data = [m3]. And the next state may be
    1) near 2m, m3 -> m2
    2) idle, m3 -> silent or other
    '''
    if not (feature in state.data):
        state.data.append(feature)

    state.data_count[feature] += 1

    if state.data_count[Label.silent] > SILENT_NUM_THRESH:  # m3 -> silent
        return Output.silent
    elif state.data_count[Label.m2] > 2 and state.data_count[Label.m3] > 0 and state.data.index(Label.m3) < state.data.index(Label.m2):  # m3 -> m2
        return Output.near_2m
    else:
        pass

    return Ret.EMPTY


def state_left_action(state: State, feature: int) -> int:
    '''
    The current state is left, data = [left]. And the next state may be
    1) away 2m, left -> m1 -> m2
    2) silent, left -> silent
    '''
    if not (feature in state.data):
        state.data.append(feature)

    state.data_count[feature] += 1

    if state.data_count[Label.silent] > SILENT_NUM_THRESH:  # left -> silent
        return Output.silent
    elif state.data_count[Label.m2] > 2 and state.data_count[Label.m1] > 0 and state.data.index(Label.m1) < state.data.index(Label.m2):  # left -> m1 -> m2
        return Output.away_2m
    elif state.data_count[Label.m3] > 2 and state.data_count[Label.m2] > 0 and state.data.index(Label.m2) < state.data.index(Label.m3):  # left -> m2 -> m3
        return Output.away_3m
    # elif state.data_count[Label.m1] > 2 and state.data_count[Label.m2] > 0 and state.data.index(Label.m2) < state.data.index(Label.m1):  # left -> m2 -> m1
    #     return Output.near_1m
    else:
        pass

    return Ret.EMPTY


def state_right_action(state: State, feature: int) -> int:
    '''
    The current state is right, data = [right]. And the next state may be
    1) away 2m, right -> m3 -> m2
    2) silent, right -> silent
    '''
    if not (feature in state.data):
        state.data.append(feature)

    state.data_count[feature] += 1

    if state.data_count[Label.silent] > SILENT_NUM_THRESH:  # right -> silent
        return Output.silent
    elif state.data_count[Label.m2] > 2 and state.data_count[Label.m1] > 0 and state.data.index(Label.m1) < state.data.index(Label.m2):  # right -> m1 -> m2
        return Output.away_2m
    elif state.data_count[Label.m3] > 2 and state.data_count[Label.m2] > 0 and state.data.index(Label.m2) < state.data.index(Label.m3):  # right -> m2 -> m3
        return Output.away_3m
    # elif state.data_count[Label.m1] > 2 and state.data_count[Label.m2] > 0 and state.data.index(Label.m2) < state.data.index(Label.m1):  # right -> m2 -> m1
    #     return Output.near_1m
    else:
        pass

    return Ret.EMPTY


class ProximityDetectionFSM:
    def __init__(self):
        # define all states
        state_idle = State()
        state_near_3m = State()
        state_near_2m = State()
        state_near_1m = State()
        state_away_2m = State()
        state_away_3m = State()
        state_left = State()
        state_right = State()

        # initialize state
        # # idle state (class -1)
        state_idle.name = Output.silent
        state_idle.data_init = state_idle_data_init
        state_idle_data_init(state_idle)
        state_idle.action = state_idle_action
        state_idle.num_transition = 6
        state_idle.transitions = [Transition() for _ in range(state_idle.num_transition)]
        # transition to state_near_3m
        state_idle.transitions[0].condition = Output.near_3m
        state_idle.transitions[0].next_state = state_near_3m
        # transition to state_near_1m
        state_idle.transitions[1].condition = Output.near_1m
        state_idle.transitions[1].next_state = state_near_1m
        # transition to state_away_2m
        state_idle.transitions[2].condition = Output.away_2m
        state_idle.transitions[2].next_state = state_away_2m
        # transition to state_away_3m
        state_idle.transitions[3].condition = Output.away_3m
        state_idle.transitions[3].next_state = state_away_3m
        # transition to state_left
        state_idle.transitions[4].condition = Output.left
        state_idle.transitions[4].next_state = state_left
        # transition to state_right
        state_idle.transitions[5].condition = Output.right
        state_idle.transitions[5].next_state = state_right

        # # near 3m state
        state_near_3m.name = Output.near_3m
        state_near_3m.data_init = state_near_3m_data_init
        state_near_3m_data_init(state_near_3m)
        state_near_3m.action = state_near_3m_action
        state_near_3m.num_transition = 2
        state_near_3m.transitions = [Transition() for _ in range(state_near_3m.num_transition)]
        # transition to state_idle
        state_near_3m.transitions[0].condition = Output.silent
        state_near_3m.transitions[0].next_state = state_idle
        # transition to state_near_2m
        state_near_3m.transitions[1].condition = Output.near_2m
        state_near_3m.transitions[1].next_state = state_near_2m

        # # near 2m state
        state_near_2m.name = Output.near_2m
        state_near_2m.data_init = state_near_2m_data_init
        state_near_2m_data_init(state_near_2m)
        state_near_2m.action = state_near_2m_action
        state_near_2m.num_transition = 3
        state_near_2m.transitions = [Transition() for _ in range(state_near_2m.num_transition)]
        # transition to state_idle
        state_near_2m.transitions[0].condition = Output.silent
        state_near_2m.transitions[0].next_state = state_idle
        # transition to state_near_1m
        state_near_2m.transitions[1].condition = Output.near_1m
        state_near_2m.transitions[1].next_state = state_near_1m
        # transition to state_away_3m
        state_near_2m.transitions[2].condition = Output.away_3m
        state_near_2m.transitions[2].next_state = state_away_3m

        # # near 1m state
        state_near_1m.name = Output.near_1m
        state_near_1m.data_init = state_near_1m_data_init
        state_near_1m_data_init(state_near_1m)
        state_near_1m.action = state_near_1m_action
        state_near_1m.num_transition = 2
        state_near_1m.transitions = [Transition() for _ in range(state_near_1m.num_transition)]
        # transition to state_idle
        state_near_1m.transitions[0].condition = Output.silent
        state_near_1m.transitions[0].next_state = state_idle
        # transition to state_away_2m
        state_near_1m.transitions[1].condition = Output.away_2m
        state_near_1m.transitions[1].next_state = state_away_2m

        # # away 2m state
        state_away_2m.name = Output.away_2m
        state_away_2m.data_init = state_away_2m_data_init
        state_away_2m_data_init(state_away_2m)
        state_away_2m.action = state_away_2m_action
        state_away_2m.num_transition = 3
        state_away_2m.transitions = [Transition() for _ in range(state_away_2m.num_transition)]
        # transition to state_idle
        state_away_2m.transitions[0].condition = Output.silent
        state_away_2m.transitions[0].next_state = state_idle
        # transition to state_away_3m
        state_away_2m.transitions[1].condition = Output.away_3m
        state_away_2m.transitions[1].next_state = state_away_3m
        # transition to state_near_1m
        state_away_2m.transitions[2].condition = Output.near_1m
        state_away_2m.transitions[2].next_state = state_near_1m

        # # away 3m state
        state_away_3m.name = Output.away_3m
        state_away_3m.data_init = state_away_3m_data_init
        state_away_3m_data_init(state_away_3m)
        state_away_3m.action = state_away_3m_action
        state_away_3m.num_transition = 2
        state_away_3m.transitions = [Transition() for _ in range(state_away_3m.num_transition)]
        # transition to state_idle
        state_away_3m.transitions[0].condition = Output.silent
        state_away_3m.transitions[0].next_state = state_idle
        # transition to state_near_2m
        state_away_3m.transitions[1].condition = Output.near_2m
        state_away_3m.transitions[1].next_state = state_near_2m

        # # left state
        state_left.name = Output.left
        state_left.data_init = state_left_data_init
        state_left_data_init(state_left)
        state_left.action = state_left_action
        state_left.num_transition = 3
        state_left.transitions = [Transition() for _ in range(state_left.num_transition)]
        # transition to silent
        state_left.transitions[0].condition = Output.silent
        state_left.transitions[0].next_state = state_idle
        # transition to away_2m
        state_left.transitions[1].condition = Output.away_2m
        state_left.transitions[1].next_state = state_away_2m
        # transition to away_3m
        state_left.transitions[2].condition = Output.away_3m
        state_left.transitions[2].next_state = state_away_3m
        # # transition to near_1m
        # state_left.transitions[3].condition = Output.near_1m
        # state_left.transitions[3].next_state = state_near_1m

        # # right state
        state_right.name = Output.right
        state_right.data_init = state_right_data_init
        state_right_data_init(state_right)
        state_right.action = state_right_action
        state_right.num_transition = 3
        state_right.transitions = [Transition() for _ in range(state_right.num_transition)]
        # transition to silent
        state_right.transitions[0].condition = Output.silent
        state_right.transitions[0].next_state = state_idle
        # transition to away_2m
        state_right.transitions[1].condition = Output.away_2m
        state_right.transitions[1].next_state = state_away_2m
        # transition to away_3m
        state_right.transitions[2].condition = Output.away_3m
        state_right.transitions[2].next_state = state_away_3m
        # # transition to near_1m
        # state_right.transitions[3].condition = Output.near_1m
        # state_right.transitions[3].next_state = state_near_1m

        self.fsm = StateMachine()
        state_machine_init(self.fsm, state_idle)

    def __call__(self, feature: int) -> int:
        return state_machine_handle_event(self.fsm, feature)