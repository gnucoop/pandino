from datachat.engine_factory import create_engine
from datachat.engine_interface import DataChatEngine


# Dictionary of active agents associated to an Api Key
activeEngines: dict[str, DataChatEngine] = {}


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

