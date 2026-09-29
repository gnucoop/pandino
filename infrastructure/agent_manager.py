import threading

from datachat.engine_factory import create_engine
from datachat.engine_interface import DataChatEngine


# Dictionary of active agents associated to an Api Key
activeEngines: dict[str, DataChatEngine] = {}

# Engines with a /datachat run in progress, keyed by id(engine) rather than by
# Api Key or by the engine itself: SmolagentsEngine is a dataclass, so it is
# unhashable and compares by fields. Holding the engine as the value keeps its
# id from being reused while the run is active.
_busyEngines: dict[int, DataChatEngine] = {}
_busyLock = threading.Lock()


# Retrieves an active agent associated with an Api Key
def getAgent(api_key) -> DataChatEngine | None:
    if not api_key:
        return None
    return activeEngines.get(str(api_key))



# Retrieves an active agents or creates a new one, adding it to the activeAgents dictionary.
def createAgent(api_key, data, llm, user_name, engine_type: str, open_charts=False) -> DataChatEngine | None:
    key = str(api_key)
    if activeEngines.get(key):
        return activeEngines.get(key)

    engine = create_engine(
        engine_type= engine_type,
        api_key=key,
        user_name=user_name,
        llm=llm,
        data=data,
        open_charts=open_charts,
    )
    activeEngines[key] = engine
    return engine


# Deletes an agent from active agents.
def deleteAgent(api_key, user_name) -> DataChatEngine | None:
    key = str(api_key)
    engine = activeEngines.get(key)
    if not api_key or not engine or not user_name:
        return None

    engine.close()
    return activeEngines.pop(key)


# Lists all active agents
def listAgents():
    return {k: "engine_active" for k in activeEngines.keys()}


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
