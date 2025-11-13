import json
import logging
import time
import requests
from pydantic import BaseModel

# Constants
MAX_RETRIES = 10
RETRY_INTERVAL_MS = 1000

logger = logging.getLogger(__name__)


class CopilotMessage(BaseModel):
    """Model for copilot chat messages."""
    message: str
    user_id: str
    watermark: str | None = None


def copilot_chat(
    item: CopilotMessage,
    copilot_token_secret: str,
    copilot_conversation_url: str
) -> dict:
    """
    Send a message to Copilot using Direct Line API and retrieve the response.
    
    Args:
        item: CopilotMessage containing the message, user_id, and optional watermark
        copilot_token_secret: Direct Line secret for authentication
        copilot_conversation_url: Base URL for Direct Line conversations API
        
    Returns:
        dict with 'response' (str) and 'last_watermark' (str) keys
        
    Raises:
        requests.exceptions.RequestException: If API calls fail
        ValueError: If required fields are missing from API responses
        KeyError: If expected keys are not found in API responses
    """
    # Use the secret directly to start conversation (skip token generation step)
    # The Direct Line secret from App Config can be used directly
    logger.info("Using Direct Line secret directly to start conversation")
    
    # Start conversation
    # POST https://directline.botframework.com/v3/directline/conversations
    
    headers = {
        'Authorization': f"Bearer {copilot_token_secret}"
    }
    
    try:
        logger.info("Starting copilot conversation")
        conversation_request = requests.post(copilot_conversation_url, headers=headers)
        conversation_request.raise_for_status()
        
        conversation_results = conversation_request.json()
        logger.info(f"Conversation started successfully")
        
        if 'conversationId' not in conversation_results:
            logger.error(f"conversationId not found in response. Full response: {conversation_results}")
            raise ValueError(f"conversationId not found in response: {conversation_results}")
            
        conversation_id = conversation_results['conversationId']
        conversation_token = conversation_results['token']
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in start conversation API call: {e}", stack_info=True, exc_info=True)
        raise
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing conversation response: {e}", stack_info=True, exc_info=True)
        raise

    # Send activity
    conversation_url = f"{copilot_conversation_url}/{conversation_id}/activities"

    headers = {
        'Authorization': f"Bearer {conversation_token}",
        'Content-Type': 'application/json'
    }
    
    payload = {
        "locale": "en-EN",
        "type": "message",
        "text": f"{item.message}",
        "from": {
            "id": "user1"
        }
    }

    data = json.dumps(payload)

    try:
        send_activity_request = requests.post(conversation_url, headers=headers, data=data)
        send_activity_request.raise_for_status()
        send_activity_results = send_activity_request.json()
        activity_id = send_activity_results['id']
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in Send Activity API call: {e}", stack_info=True, exc_info=True)
        raise
    except KeyError as e:
        logger.error(f"Error parsing send activity response: {e}. Response: {send_activity_results}", 
                    stack_info=True, exc_info=True)
        raise

    # Receive activity
    # Append ?watermark=<value> if watermark is provided so that response to follow up question is retrieved
    if item.watermark is not None:
        conversation_url += f"?watermark={item.watermark}"

    # GET https://directline.botframework.com/v3/directline/conversations/abc123/activities?watermark=0001a-94
    headers = {
        'Authorization': f"Bearer {conversation_token}"
    }

    copilot_response = "No valid response found in activities"
    watermark = None

    for attempt in range(MAX_RETRIES):
        try:
            get_activity_request = requests.get(conversation_url, headers=headers)
            get_activity_request.raise_for_status()  # Raise an exception for HTTP errors
            get_activity_results = get_activity_request.json()  # Parse JSON response
            
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
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in getting messages API: {e}", stack_info=True, exc_info=True)
            copilot_response = "No response from Copilot within the configured timeout limit"
            get_activity_results = None
        except Exception as e:
            logger.error(f"Unexpected error in getting messages: {e}", stack_info=True, exc_info=True)
            copilot_response = "No response from Copilot within the configured timeout limit"
            get_activity_results = None
            
        time.sleep(RETRY_INTERVAL_MS / 1000.0)  # Convert milliseconds to seconds
    
    logger.info(f"Copilot response: {copilot_response}")

    return {"response": copilot_response, "last_watermark": watermark}
