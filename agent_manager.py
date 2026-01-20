from datachat.pandasai_engine import PandasAIEngine
from flask import jsonify
import os
import shutil

# Dictionary of active agents associated to an Api Key
activeEngines: dict[str, PandasAIEngine] = {}


# Retrieves an active agent associated with an Api Key
def getAgent(api_key) -> PandasAIEngine | None:
    if not api_key:
        return None
    return activeEngines.get(str(api_key))



# Retrieves an active agents or creates a new one, adding it to the activeAgents dictionary.
def createAgent(api_key, data, llm, user_name, open_charts=False) -> PandasAIEngine | None:
    key = str(api_key)
    if activeEngines.get(key):
        return activeEngines.get(key)

    engine = PandasAIEngine(
        api_key=key,
        user_name=user_name,
        llm=llm,
        data=data,
        open_charts=open_charts,
    )
    activeEngines[key] = engine
    return engine


# Deletes an agent from active agents.
def deleteAgent(api_key, user_name) -> PandasAIEngine | None:
    key = str(api_key)
    engine = activeEngines.get(key)
    if not api_key or not engine or not user_name:
        return None

    engine.close()
    return activeEngines.pop(key)


# Lists all active agents
def listAgents():
    return {k: "engine_active" for k in activeEngines.keys()}

