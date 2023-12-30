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


class Label(IntEnum):
    silent = -1
    l0 = 0
    l1 = 1
    l2 = 2
    l3 = 3
    l4 = 4


class Output(IntEnum):
    silent = 0
    activate = 1
    ready = 2
    slide_left = 3
    slide_right = 4
    slide_up = 5
    slide_down = 6
    clockwise = 7
    anti_clockwise = 8


ACTIVATE_THRESH = 15
ACTIVATE_ADD_NUM = 2

DEACTIVATE_THRESH = 0
DEACTIVATE_ADD_NUM = -1

READY_THRESH = 7
READY_MAX_THRESH = 8
READY_ADD_NUM_LARGE = 1
READY_ADD_NUM_LITTLE = 0.5

UNREADY_THRESH = 6
UNREADY_ADD_NUM = -0.5

CLOCKWISE_THRESH = 3
CLOCKWISE_CONTINUE_THRESH = 2
ANTI_CLOCKWISE_THRESH = -3
ANTI_CLOCKWISE_CONTINUE_THRESH = -2

CLOCKWISE_ADD_NUM_MAP = {
    '14': 1, '42': 1, '23': 1, '31': 1,
    '13': -1, '32': -1, '24': -1, '41': -1,
    '01': 0, '04': 0, '02': 0, '03': 0, '30': 0, '20': 0, '40': 0, '10': 0,
}
CLOCKWISE_ABSENT_ADD_NUM = -1
ANTI_CLOCKWISE_ABSENT_ADD_NUM = 1

SLIDE_THRESH = 2
FROZEN_TIME = 5


def state_idle_data_init(state: State):
    state.data: int = 0  # active_socre


def state_activate_data_init(state: State):
    # [activate_score, ready_score, clockwise_score, last_feature, cur_feature]
    state.data: List[int] = [ACTIVATE_THRESH, 0, 0, -1, -1]


def state_ready_data_init(state: State):
    # [ready_score, slide_score_up, slide_score_down, slide_score_left, slide_score_right]
    state.data: List[int] = [READY_THRESH, 0, 0, 0, 0]


def state_slide_data_init(state: State):
    state.data: int = FROZEN_TIME


def state_clockwise_data_init(state: State):
    # [clockwise_score, last_feature, cur_feature]
    state.data: List[int] = [CLOCKWISE_THRESH, -1, -1]


def state_anti_clockwise_data_init(state: State):
    # [anti_clockwise_score, last_feature, cur_feature]
    state.data: List[int] = [ANTI_CLOCKWISE_THRESH, -1, -1]


def state_idle_action(state: State, feature: int) -> int:
    if feature in [Label.l0, Label.l1]:
        state.data += ACTIVATE_ADD_NUM

    if state.data >= ACTIVATE_THRESH:
        return Output.activate

    return Ret.EMPTY


def state_activate_action(state: State, feature: int) -> int:
    # update current feature
    cur_feature = state.data[-1]
    state.data[-1] = feature

    if Label.silent == feature:
        state.data[0] += DEACTIVATE_ADD_NUM  # activate_score

        if state.data[0] <= DEACTIVATE_THRESH:
            return Output.silent
    else:
        state.data[0] = ACTIVATE_THRESH

    if Label.l0 == feature:
        if state.data[1] >= READY_THRESH:  # ready_score
            state.data[1] += READY_ADD_NUM_LITTLE
        else:
            state.data[1] += READY_ADD_NUM_LARGE

        if state.data[1] >= READY_THRESH:
            return Output.ready
    else:
        state.data[1] += UNREADY_ADD_NUM

        if state.data[1] < UNREADY_THRESH:
            if feature in [Label.l1, Label.l2, Label.l3, Label.l4] and cur_feature == feature:
                return Output.silent

            # check clockwise score map
            ff = f'{state.data[-2]}{state.data[-1]}'

            if state.data[2] > 0:  # clockwise_score
                state.data[2] += CLOCKWISE_ADD_NUM_MAP.get(ff, CLOCKWISE_ABSENT_ADD_NUM)
            elif state.data[2] < 0:
                state.data[2] += CLOCKWISE_ADD_NUM_MAP.get(ff, ANTI_CLOCKWISE_ABSENT_ADD_NUM)
            else:
                state.data[2] += CLOCKWISE_ADD_NUM_MAP.get(ff, 0)

            if state.data[2] >= CLOCKWISE_THRESH and state.data[-1] in [Label.l1, Label.l2, Label.l3, Label.l4]:
                return Output.clockwise
            if state.data[2] <= ANTI_CLOCKWISE_THRESH and state.data[-1] in [Label.l1, Label.l2, Label.l3, Label.l4]:
                return Output.anti_clockwise

    # update last feature
    state.data[-2] = feature

    return Ret.EMPTY


def state_ready_action(state: State, feature: int) -> int:
    if Label.l0 == feature:
        if state.data[0] >= READY_THRESH:
            state.data[0] += READY_ADD_NUM_LITTLE
            state.data[0] = min(READY_MAX_THRESH, state.data[0])
    else:
        state.data[0] += UNREADY_ADD_NUM

        if state.data[0] < UNREADY_THRESH:
            return Output.activate

        if Label.l1 == feature:
            state.data[1] += 1  # slide_score_up
            if state.data[1] >= SLIDE_THRESH:
                return Output.slide_up
        elif Label.l2 == feature:
            state.data[2] += 1  # slide_score_down
            if state.data[2] >= SLIDE_THRESH:
                return Output.slide_down
        elif Label.l3 == feature:
            state.data[3] += 1  # slide_score_left
            if state.data[3] >= SLIDE_THRESH:
                return Output.slide_left
        elif Label.l4 == feature:
            state.data[4] += 1  # slide_score_right
            if state.data[4] >= SLIDE_THRESH:
                return Output.slide_right
        else:
            pass

    return Ret.EMPTY


def state_slide_action(state: State, feature: int) -> int:
    state.data -= 1

    if 0 == state.data:
        return Output.activate

    return Ret.EMPTY


def state_clockwise_action(state: State, feature: int) -> int:
    if feature in [Label.l1, Label.l2, Label.l3, Label.l4] and state.data[-1] == feature:
        return Ret.EMPTY

    state.data[-1] = feature

    # check clockwise score map
    ff = f'{state.data[-2]}{state.data[-1]}'
    state.data[0] += CLOCKWISE_ADD_NUM_MAP.get(ff, CLOCKWISE_ABSENT_ADD_NUM)

    state.data[0] = min(CLOCKWISE_THRESH, state.data[0])

    state.data[-2] = feature

    if state.data[0] < CLOCKWISE_CONTINUE_THRESH:
        return Output.activate
    else:
        if state.data[-1] in [Label.l1, Label.l2, Label.l3, Label.l4]:
            return Output.clockwise
        else:
            return Ret.EMPTY


def state_anti_clockwise_action(state: State, feature: int) -> int:
    if feature in [Label.l1, Label.l2, Label.l3, Label.l4] and state.data[-1] == feature:
        return Ret.EMPTY

    state.data[-1] = feature

    # check clockwise score map
    ff = f'{state.data[-2]}{state.data[-1]}'
    state.data[0] += CLOCKWISE_ADD_NUM_MAP.get(ff, ANTI_CLOCKWISE_ABSENT_ADD_NUM)

    state.data[0] = max(ANTI_CLOCKWISE_THRESH, state.data[0])

    state.data[-2] = feature

    if state.data[0] > ANTI_CLOCKWISE_CONTINUE_THRESH:
        return Output.activate
    else:
        if state.data[-1] in [Label.l1, Label.l2, Label.l3, Label.l4]:
            return Output.anti_clockwise
        else:
            return Ret.EMPTY


class GestureControlFSM:
    def __init__(self):
        # define all states
        state_idle = State()
        state_activate = State()
        state_ready = State()
        state_slide_up = State()
        state_slide_down = State()
        state_slide_left = State()
        state_slide_right = State()
        state_clockwise = State()
        state_anti_clockwise = State()

        # initialize state
        # # idle state
        state_idle.name = Output.silent
        state_idle.data_init = state_idle_data_init
        state_idle_data_init(state_idle)
        state_idle.action = state_idle_action
        state_idle.num_transition = 1
        state_idle.transitions = [Transition() for _ in range(state_idle.num_transition)]
        # transition to state_activate
        state_idle.transitions[0].condition = Output.activate
        state_idle.transitions[0].next_state = state_activate

        # # activate state
        state_activate.name = Output.activate
        state_activate.data_init = state_activate_data_init
        state_activate_data_init(state_activate)
        state_activate.action = state_activate_action
        state_activate.num_transition = 4
        state_activate.transitions = [Transition() for _ in range(state_activate.num_transition)]
        # transition to state_idle
        state_activate.transitions[0].condition = Output.silent
        state_activate.transitions[0].next_state = state_idle
        # transition to state_ready
        state_activate.transitions[1].condition = Output.ready
        state_activate.transitions[1].next_state = state_ready
        # transition to state_clockwise
        state_activate.transitions[2].condition = Output.clockwise
        state_activate.transitions[2].next_state = state_clockwise
        # transition to state_anti_clockwise
        state_activate.transitions[3].condition = Output.anti_clockwise
        state_activate.transitions[3].next_state = state_anti_clockwise

        # # ready state
        state_ready.name = Output.ready
        state_ready.data_init = state_ready_data_init
        state_ready_data_init(state_ready)
        state_ready.action = state_ready_action
        state_ready.num_transition = 5
        state_ready.transitions = [Transition() for _ in range(state_ready.num_transition)]
        # transition to state_activate
        state_ready.transitions[0].condition = Output.activate
        state_ready.transitions[0].next_state = state_activate
        # transition to state_slide_up
        state_ready.transitions[1].condition = Output.slide_up
        state_ready.transitions[1].next_state = state_slide_up
        # transition to state_slide_down
        state_ready.transitions[2].condition = Output.slide_down
        state_ready.transitions[2].next_state = state_slide_down
        # transition to state_slide_left
        state_ready.transitions[3].condition = Output.slide_left
        state_ready.transitions[3].next_state = state_slide_left
        # transition to state_slide_right
        state_ready.transitions[4].condition = Output.slide_right
        state_ready.transitions[4].next_state = state_slide_right

        # # slide state up
        state_slide_up.name = Output.slide_up
        state_slide_up.data_init = state_slide_data_init
        state_slide_data_init(state_slide_up)
        state_slide_up.action = state_slide_action
        state_slide_up.num_transition = 1
        state_slide_up.transitions = [Transition() for _ in range(state_slide_up.num_transition)]
        # transition to state_activate
        state_slide_up.transitions[0].condition = Output.activate
        state_slide_up.transitions[0].next_state = state_activate

        # # slide state down
        state_slide_down.name = Output.slide_down
        state_slide_down.data_init = state_slide_data_init
        state_slide_data_init(state_slide_down)
        state_slide_down.action = state_slide_action
        state_slide_down.num_transition = 1
        state_slide_down.transitions = [Transition() for _ in range(state_slide_down.num_transition)]
        # transition to state_activate
        state_slide_down.transitions[0].condition = Output.activate
        state_slide_down.transitions[0].next_state = state_activate

        # # slide state left
        state_slide_left.name = Output.slide_left
        state_slide_left.data_init = state_slide_data_init
        state_slide_data_init(state_slide_left)
        state_slide_left.action = state_slide_action
        state_slide_left.num_transition = 1
        state_slide_left.transitions = [Transition() for _ in range(state_slide_left.num_transition)]
        # transition to state_activate
        state_slide_left.transitions[0].condition = Output.activate
        state_slide_left.transitions[0].next_state = state_activate

        # # slide state up
        state_slide_right.name = Output.slide_right
        state_slide_right.data_init = state_slide_data_init
        state_slide_data_init(state_slide_right)
        state_slide_right.action = state_slide_action
        state_slide_right.num_transition = 1
        state_slide_right.transitions = [Transition() for _ in range(state_slide_right.num_transition)]
        # transition to state_activate
        state_slide_right.transitions[0].condition = Output.activate
        state_slide_right.transitions[0].next_state = state_activate

        # # clockwise state
        state_clockwise.name = Output.clockwise
        state_clockwise.data_init = state_clockwise_data_init
        state_clockwise_data_init(state_clockwise)
        state_clockwise.action = state_clockwise_action
        state_clockwise.num_transition = 1
        state_clockwise.transitions = [Transition() for _ in range(state_clockwise.num_transition)]
        # transition to state_activate
        state_clockwise.transitions[0].condition = Output.activate
        state_clockwise.transitions[0].next_state = state_activate

        # # anti-clockwise state
        state_anti_clockwise.name = Output.anti_clockwise
        state_anti_clockwise.data_init = state_anti_clockwise_data_init
        state_anti_clockwise_data_init(state_anti_clockwise)
        state_anti_clockwise.action = state_anti_clockwise_action
        state_anti_clockwise.num_transition = 1
        state_anti_clockwise.transitions = [Transition() for _ in range(state_anti_clockwise.num_transition)]
        # transition to state_activate
        state_anti_clockwise.transitions[0].condition = Output.activate
        state_anti_clockwise.transitions[0].next_state = state_activate

        self.fsm = StateMachine()
        state_machine_init(self.fsm, state_idle)

    def __call__(self, feature: int) -> int:
        return state_machine_handle_event(self.fsm, feature)