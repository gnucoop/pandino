from pandasai import Agent
from flask import jsonify
import os
import shutil

# Dictionary of active agents associated to an Api Key
activeAgents = {}


# Retrieves an active agent associated with an Api Key
def getAgent(api_key) -> Agent | None:
    if not api_key:
        return None
    else:
        return activeAgents.get(api_key)


# Retrieves an active agents or creates a new one, adding it to the activeAgents dictionary.
def createAgent(api_key, data, llm, user_name, open_charts=False) -> Agent | None:
    string_api_key = str(api_key)
    print(f"Creating Agent: {string_api_key}")
    if activeAgents.get(string_api_key):
        print("Retrieving Agent...")
        return activeAgents.get(string_api_key)
    else:
        print("Generating new Agent...")
        agentConfig = {
            "llm": llm,
            "open_charts": open_charts,
            "save_charts": True,
            "save_charts_path": f"exports/charts/{user_name}",
            "custom_whitelisted_dependencies": ["tabulate"]
        }
        agent = Agent(data, config=agentConfig)
        activeAgents.update({string_api_key: agent})
        return agent


# Deletes an agent from active agents.
def deleteAgent(api_key, user_name) -> Agent | None:
    string_api_key = str(api_key)
    agent = activeAgents.get(string_api_key)
    if not api_key or not agent or not user_name:
        return
    elif agent:
        folderPath = f"exports/charts/{user_name}"
        if os.path.exists(folderPath):
            shutil.rmtree(folderPath, ignore_errors=True)
        else:
            print("Path does not exist")
        return activeAgents.pop(string_api_key)


# Lists all active agents
def listAgents():
    def agentConvId(agent: Agent):
        return agent.conversation_id

    activeAgentsMap = {k: agentConvId(v) for k, v in activeAgents.items()}
    print("Listing Agents...")
    return activeAgentsMap
