# evaluations/generate_eval_input.py

import json
import logging
from pathlib import Path
import sys
import time

from pydantic import BaseModel
from fastapi.testclient import TestClient
# Import the FastAPI app; ensure PYTHONPATH includes the 'src' directory
from src.main import app

from azure.search.documents import SearchClient
from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, AzureCliCredential
import requests
from appconfig import AppConfigClient

# Module-level logger and configuration variables
logger = None
copilot_token_secret = None
copilot_token_url = None
copilot_conversation_url = None
# Constants
MAX_RETRIES = 10
RETRY_INTERVAL_MS = 1000


class Item(BaseModel):
    message: str
    user_id: str
    watermark: str | None = None

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Uncomment to see DEBUG logs:
    # logging.getLogger().setLevel(logging.DEBUG)


def copilot_chat(item: Item):

    #If watermark was not provided that means that it is a new conversation. We need to perform the calls to get the auth tokens and start a conversation
    
    headers = {
    'Authorization': f"Bearer {copilot_token_secret}"
    }
    # Get auth token
    try:
        logger.info("Getting copilot token")
        r = requests.post(copilot_token_url,headers=headers)
    except:
        logger.error("Error in get token API call: %s",r.text,stack_info=True,exc_info=True)   

    token_results = r.json()
    auth_token = token_results['token']
    token_expires_in = token_results['expires_in']
    # conversation_id = token_results['conversationId']

    # Start conversation
    
    # POST https://directline.botframework.com/v3/directline/conversations
    
    headers = {
    'Authorization': f"Bearer {auth_token}"
    }
    
    try:
        logger.info("Starting copilot conversation")
        conversation_request = requests.post(copilot_conversation_url,headers=headers)
        
    except:
        logger.error("Error in start conversation API call: %s",conversation_request.text,stack_info=True,exc_info=True)   

    conversation_results = conversation_request.json()
    conversation_id = conversation_results['conversationId']
    conversation_token = conversation_results['token']

    # Send activity

    conversation_url=f"{copilot_conversation_url}/{conversation_id}/activities"

    headers = {
    'Authorization': f"Bearer {conversation_token}",
    'Content-Type': 'application/json'
    }
    
    payload =  {
        "locale": "en-EN",
        "type": "message",
        "text": f"{item.message}",
        "textformat": "plain"
    }

    data = json.dumps(payload)

    try:
        send_activity_request = requests.post(conversation_url, headers = headers, data = data)
    except:
        logger.error("Error in Send Activity API call: %s",send_activity_request.text,stack_info=True,exc_info=True)   

    send_activity_results = send_activity_request.json()
    activity_id = send_activity_results['id']

    # Receive activity
    # Append ?watermark=<value> if watermark is provided so that response to follow up question is retrieved

    if item.watermark is not None:
        conversation_url+=f"?watermark={item.watermark}"

    # GET https://directline.botframework.com/v3/directline/conversations/abc123/activities?watermark=0001a-94

    headers = {
    'Authorization': f"Bearer {conversation_token}"
    }  

    for attempt in range(MAX_RETRIES):
        try:
            get_activity_request = requests.get(conversation_url, headers = headers)
            get_activity_request.raise_for_status()  # Raise an exception for HTTP errors
            get_activity_results = get_activity_request.json()  # Parse JSON response
            
            # logger.info(f"Get Activity JSON: {get_activity_results}")
            # logger.info(f"Attempt: {attempt}")
            

            if "activities" in get_activity_results:
                for activity in get_activity_results["activities"]:
                    if activity.get("type") == "message" and activity.get("from", {}).get("role") == "bot":
                        copilot_response = activity.get("text", "No text available")
                        watermark = get_activity_results.get('watermark')
                        break
                else:
                    # Fallback if no matching activity is found
                    copilot_response = "No valid response found in activities"
                    watermark = get_activity_results.get('watermark')
            # If a valid response is found, exit the retry loop
            if copilot_response != "No valid response found in activities":
                break                    
        except:
            logger.error("Error in getting messages API: %s",r.text,stack_info=True,exc_info=True)
            copilot_response = "No response from Copilot within the configured timeout limit"   
            get_activity_results = None
        time.sleep(RETRY_INTERVAL_MS / 1000.0)  # Convert milliseconds to seconds

    
    logger.info(f"Copilot response: {copilot_response}") 

    return {"response": copilot_response, "last_watermark": watermark}

def main():
    global logger, copilot_token_secret, copilot_token_url
    
    setup_logging()
    logger = logging.getLogger("generate_eval_input")

    logger.info("🔧 Loading App Configuration settings")

    credential = ChainedTokenCredential(ManagedIdentityCredential(), AzureCliCredential())
    cfg = AppConfigClient()
    copilot_token_secret = cfg.get("COPILOT_TOKEN_SECRET_NAME")
    copilot_token_url = cfg.get("COPILOT_TOKEN_URL")
    copilot_conversation_url = cfg.get("COPILOT_CONVERSATION_URL")

    in_path = Path(__file__).parent.parent / "dataset" / "golden-dataset-copilot.jsonl"
    out_path = Path(__file__).parent.parent / "dataset" / "eval-input-copilot.jsonl"
    logger.info(f"📄 Reading queries from: {in_path}")
    logger.info(f"📝 Writing eval input to: {out_path}")

    total = sum(1 for _ in open(in_path, encoding="utf-8"))
    logger.info(f"🔢 Total entries to process: {total}")

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin, start=1):
            data = json.loads(line)
            
            query = data["query"]
            truth = data.get("ground-truth") or data.get("truth")  # adapt field name
            logger.info(f"[{idx}/{total}] ▶ Generating response for query: {query!r}")
            # Call Copilot Direct Lines APIs
            copilot_json = {
                    "message": query,
                    "user_id": "anonymous",
                    }


            copilot_item = Item(**copilot_json)
            resp = copilot_chat(copilot_item)
            response_text = resp.text
            logger.debug(f"[{idx}] Response received (first 100 chars): {response_text[:100]!r}")

            # Build record without 'item' field
            record = {
                "query": query,
                "truth": truth,
                "response": response_text,
                "context": "N/A"
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"[{idx}/{total}] ✔ Record written")

    logger.info("✅ Eval input generation complete.")

if __name__ == "__main__":
    main()
