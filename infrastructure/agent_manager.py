import threading
from typing import Any, Callable

from datachat.engine_factory import create_engine
from datachat.engine_interface import DataChatEngine

# Agent kinds sharing the registry. A user may hold one session of each kind
# at the same time under the same Api Key, so the kind is part of the key.
DATACHAT = "datachat"
INTERVIEWER = "interviewer"

# Engines with a /datachat run in progress, keyed by id(engine) rather than by
# Api Key or by the engine itself: SmolagentsEngine is a dataclass, so it is
# unhashable and compares by fields. Holding the engine as the value keeps its
# id from being reused while the run is active.
_busyEngines: dict[int, DataChatEngine] = {}
_busyLock = threading.Lock()


# Dictionary of active agents associated to an (agent kind, Api Key) pair
activeEngines: dict[tuple[str, str], Any] = {}

# Guards every write to activeEngines, so a delete can check which engine is
# registered and remove it as one step. Factories and close() run outside it.
_registryLock = threading.Lock()


def _get(kind: str, api_key) -> Any | None:
    if not api_key:
        return None
    return activeEngines.get((kind, str(api_key)))


def _get_or_create(kind: str, api_key, factory: Callable[[], Any]) -> Any:
    key = (kind, str(api_key))
    if activeEngines.get(key):
        return activeEngines.get(key)

    engine = factory()
    with _registryLock:
        activeEngines[key] = engine
    return engine


def _delete(kind: str, api_key, user_name) -> Any | None:
    key = (kind, str(api_key))
    if not api_key or not user_name:
        return None

    with _registryLock:
        engine = activeEngines.pop(key, None)
    if not engine:
        return None

    engine.close()
    return engine


# Retrieves an active agent associated with an Api Key
def getAgent(api_key) -> DataChatEngine | None:
    return _get(DATACHAT, api_key)


# Retrieves an active agents or creates a new one, adding it to the activeAgents dictionary.
def createAgent(api_key, data, llm, user_name, engine_type: str, open_charts=False) -> DataChatEngine | None:
    key = str(api_key)
    return _get_or_create(
        DATACHAT,
        key,
        lambda: create_engine(
            engine_type=engine_type,
            api_key=key,
            user_name=user_name,
            llm=llm,
            data=data,
            open_charts=open_charts,
        ),
    )


# Deletes an agent from active agents.
def deleteAgent(api_key, user_name) -> DataChatEngine | None:
    return _delete(DATACHAT, api_key, user_name)


# Retrieves the active interviewer associated with an Api Key
def getInterviewer(api_key) -> Any | None:
    return _get(INTERVIEWER, api_key)


# Retrieves the active interviewer or creates one with the given factory.
def createInterviewer(api_key, factory: Callable[[], Any]) -> Any:
    return _get_or_create(INTERVIEWER, api_key, factory)


# Deletes the interviewer from active agents.
def deleteInterviewer(api_key, user_name) -> Any | None:
    return _delete(INTERVIEWER, api_key, user_name)


# Deletes the interviewer only if `engine` is still the one registered, so a
# request holding a stale engine never removes a newer session for the key.
def deleteInterviewerIfCurrent(api_key, engine) -> bool:
    if not api_key or engine is None:
        return False

    key = (INTERVIEWER, str(api_key))
    with _registryLock:
        if activeEngines.get(key) is not engine:
            return False
        activeEngines.pop(key)

    engine.close()
    return True


# Lists all active agents
def listAgents():
    return {
            (api_key if kind == DATACHAT else f"{kind}:{api_key}"): "engine_active"
            for kind, api_key in activeEngines.keys()
        }

# Marks an engine as running without waiting. Returns False if it is already busy.
def try_acquire_run(engine: DataChatEngine) -> bool:
    with _busyLock:
        if id(engine) in _busyEngines:
            return False
        _busyEngines[id(engine)] = engine
        return True


# Clears the running mark of an engine. Safe to call from a finally block.
def release_run(engine: DataChatEngine) -> None:
    with _busyLock:
        _busyEngines.pop(id(engine), None)
    
