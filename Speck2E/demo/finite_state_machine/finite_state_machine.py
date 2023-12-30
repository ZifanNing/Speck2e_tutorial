from enum import IntEnum
import logging
from typing import Any, Callable, Optional


logger = logging.getLogger()


class Ret(IntEnum):
    EMPTY = -1
    NO_MATCHED_TRANSITION = -2


class Transition:
    def __init__(self):
        self.condition: int = 0  # the condition that transit one state to another state
        self.next_state: Optional[State] = None  #transit to the state, a instance of class State


class State:
    def __init__(self):
        self.name: int = 0  # the state name, should be the final prediction of your model.
        self.data: Any = None  # maybe a container for predictions from chip.
        self.data_count: Any = None  # maybe a container for counting predictions from chip.
        self.data_init: Callable = None  # a function for initializing data
        self.action: Callable = None  # a function for checking whether transit from one state to another state.
        self.num_transition: int = 0  # the num of transitions that the state to other states
        self.transitions: Optional[Transition] = None  # a container for each transition


class StateMachine:
    def __init__(self):
        # the current state
        self.current_state: Optional[State] = None
        # the previous state
        self.previous_state: Optional[State] = None


def state_machine_init(fsm: StateMachine, initial_state: State):
    '''
    Initialize the state machine.
    '''
    if fsm is None:
        logger.error(f'Initialize fsm. fsm cannot be None.')
        return

    fsm.current_state = initial_state
    fsm.previous_state = None


def get_transition(fsm: StateMachine, data: int) -> Optional[Transition]:
    '''
    Check whether transit state or not.
    '''
    for tran_idx in range(fsm.current_state.num_transition):
        transition = fsm.current_state.transitions[tran_idx]
        if transition.condition == data:
            return transition

    return Ret.NO_MATCHED_TRANSITION


def state_machine_handle_event(fsm: StateMachine, feature: int) -> int:
    '''
    According to the feature which is one prediction from chip, check that whether transit current state to another state.
    '''
    ret = fsm.current_state.action(fsm.current_state, feature)
    # logger.info(
    #     f'ret: {ret}, feature: {feature}, current state name: {fsm.current_state.name}, '
    #     f'data: {fsm.current_state.data}, data_count: {fsm.current_state.data_count}'
    # )

    if Ret.EMPTY == ret:
        return ret
    else:
        transition = get_transition(fsm, ret)

        if Ret.NO_MATCHED_TRANSITION == transition:
            # logger.warning(
            #     f'Cannot match transition! current state name: {fsm.current_state.name}, '
            #     f'data: {fsm.current_state.data}, data_count: {fsm.current_state.data_count}, action ret: {ret}'
            # )
            return ret
        else:
            fsm.current_state.data_init(fsm.current_state)

            fsm.previous_state = fsm.current_state
            fsm.current_state = transition.next_state
            return fsm.current_state.name