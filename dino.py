import json
import requests

def dino_authenticate(graphql_url, auth_token):
    req_body = json.dumps({"query": "{form_data(limit:1){id}}", "variables": {}})
    response = requests.post(graphql_url, data=req_body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + auth_token})
    if response.status_code != 200:
        return f"Graphql query not ok, status: {response.status_code}"
    try:
        result = response.json()
        if "errors" in result:
            return f"Error authenticating dino: {result['errors'][0]['message']}"
        return None
    except json.JSONDecodeError as e:
        return f"Error decoding dino response: {e}"
    
