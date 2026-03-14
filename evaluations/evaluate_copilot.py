import sys
import logging
import os
from pathlib import Path
from datetime import datetime, UTC
import asyncio

from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, AzureCliCredential

import azure.ai.projects
print(azure.ai.projects.__version__)

from azure.ai.projects import AIProjectClient

from azure.core.exceptions import HttpResponseError
from azure.ai.projects.models import (
    EvaluatorConfiguration,
    EvaluatorIds,
    Evaluation,
    InputDataset
)
from appconfig import AppConfigClient
from keyvault import KeyVaultClient
from azure.ai.evaluation.red_team import RedTeam, RiskCategory, AttackStrategy
from copilot_client import copilot_chat, CopilotMessage

# Suppress Azure SDK HTTP logging
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

# 1) Configuration & logging

cfg = AppConfigClient()
model_api_key_name = cfg.get("EVALUATIONS_MODEL_API_KEY_SECRET_NAME") or "evaluationsModelApiKey"
keyvault_client = KeyVaultClient()
MODEL_API_KEY = keyvault_client.get_secret(model_api_key_name)
if not MODEL_API_KEY:
    logger = logging.getLogger("cloud_evaluation")
    logger.error(f"Model API key secret '{model_api_key_name}' not found in Key Vault!")
    sys.exit(1)

PROJECT_ENDPOINT      = cfg.get("AI_FOUNDRY_PROJECT_ENDPOINT")
MODEL_ENDPOINT        = cfg.get("AI_FOUNDRY_ACCOUNT_ENDPOINT")  # e.g. https://<account>.services.ai.azure.com
MODEL_DEPLOYMENT_NAME = cfg.get("CHAT_DEPLOYMENT_NAME")
DATASET_NAME          = cfg.get("COPILOT_DATASET_NAME", "copilot-eval-dataset")
DATASET_VERSION       = datetime.now(UTC).strftime("v%Y%m%d%H%M%S")
INPUT_FILE            = cfg.get(
    "EVAL_INPUT_FILE",
    str(Path(__file__).parent.parent / "dataset" / "eval-input-copilot.jsonl")
)
AZURE_AI_PROJECT      = cfg.get("AI_FOUNDRY_PROJECT_ENDPOINT")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cloud_evaluation")

# Validate necessary config
if not (PROJECT_ENDPOINT and MODEL_ENDPOINT and MODEL_DEPLOYMENT_NAME and MODEL_API_KEY):
    logger.error("Missing one or more required settings: PROJECT_ENDPOINT, MODEL_ENDPOINT, CHAT_DEPLOYMENT_NAME, or MODEL_API_KEY")
    sys.exit(1)

# 2) Initialize AIProjectClient
credential = ChainedTokenCredential(ManagedIdentityCredential(), AzureCliCredential())
project_client = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential)
logger.info(f"Connected to AI Foundry: {PROJECT_ENDPOINT}")

# 3) Upload dataset
logger.info(f"Uploading dataset '{DATASET_NAME}' version '{DATASET_VERSION}'")
try:
    dataset = project_client.datasets.upload_file(
        name=DATASET_NAME,
        version=DATASET_VERSION,
        file_path=INPUT_FILE
    )
    logger.info(f"Dataset uploaded: {dataset.id}")
except HttpResponseError as e:
    logger.error(f"Dataset upload failed: {e}")
    sys.exit(1)

# 4) Configure evaluators
evaluators = {
    "completeness": EvaluatorConfiguration(
        id=EvaluatorIds.RESPONSE_COMPLETENESS,
        init_params={"deployment_name": MODEL_DEPLOYMENT_NAME, "threshold": 4 },
        data_mapping={"response": "${data.response}", "ground_truth": "${data.truth}"}
    ),
    "relevance": EvaluatorConfiguration(
        id=EvaluatorIds.RELEVANCE,
        init_params={"deployment_name": MODEL_DEPLOYMENT_NAME},
        data_mapping={"response": "${data.response}", "query": "${data.query}"}
    ),
    # "retrieval": EvaluatorConfiguration(
    #     id=EvaluatorIds.RETRIEVAL,
    #     init_params={"deployment_name": MODEL_DEPLOYMENT_NAME},
    #     data_mapping={"query": "${data.query}", "context": "${data.context}"}
    # ),
    "coherence": EvaluatorConfiguration(
        id=EvaluatorIds.COHERENCE,
        init_params={"deployment_name": MODEL_DEPLOYMENT_NAME},
        data_mapping={"query": "${data.query}", "response": "${data.response}"}
    ),
    # "similarity": EvaluatorConfiguration(
    #     id=EvaluatorIds.SIMILARITY,
    #     init_params={"deployment_name": MODEL_DEPLOYMENT_NAME},
    #     data_mapping={"query": "${data.query}", "context": "${data.context}","ground_truth": "${data.truth}"}
    # ),    
    # "groundedness": EvaluatorConfiguration(
    #     id=EvaluatorIds.GROUNDEDNESS,
    #     init_params={"deployment_name": MODEL_DEPLOYMENT_NAME},
    #     data_mapping={"query": "${data.query}", "context": "${data.context}","response": "${data.response}"}
    # ),    
    "safety": EvaluatorConfiguration(
        id=EvaluatorIds.CONTENT_SAFETY,
        init_params={"deployment_name": MODEL_DEPLOYMENT_NAME},        
        data_mapping={"response": "${data.response}", "query": "${data.query}"}
    )
}

# 5) Submit evaluation with both required headers
# Get commit ID from environment variable or command-line argument
commit_id = os.getenv("COMMIT_ID") or (sys.argv[1] if len(sys.argv) > 1 else "unknown")
eval_timestamp = datetime.now(UTC).strftime("%Y%m%d-%H:%M")
display_name = f"Auto evaluation-CopilotStudio-{eval_timestamp}"

evaluation = Evaluation(
    display_name=display_name,
    description="Pre-deployment RAG evaluation",
    data=InputDataset(id=dataset.id),
    evaluators=evaluators
)

logger.info("Submitting evaluation with required headers...")
try:
    eval_response = project_client.evaluations.create(
        evaluation,
        headers = {
            "model-endpoint": MODEL_ENDPOINT,
            "api-key": MODEL_API_KEY,
        }
    )
    logger.info(f"Evaluation created: {eval_response.name} (status: {eval_response.status})")
except HttpResponseError as e:
    logger.error(f"Evaluation submission failed: {e}")
    sys.exit(1)

# 6) Save run details
try:
    # Attempt to extract the AiStudioEvaluationUri from the response
    evaluation_url = eval_response.as_dict().get("properties", {}).get("AiStudioEvaluationUri")
    if evaluation_url:
        logger.info(f"Evaluation started. You can view the results at: \033[94m\033[4m{evaluation_url}\033[0m")
    else:
        logger.warning("Evaluation started, but the evaluation URL could not be retrieved.")
except Exception as e:
    logger.error(f"An error occurred while processing the evaluation response: {e}")

## Red teaming ##

async def run_red_team_scan():
    logger.info(f"### Red Teaming Scan Starting ###")

    # Get Copilot configuration
    copilot_token_secret = cfg.get("COPILOT_TOKEN_SECRET")
    copilot_conversation_url = cfg.get("COPILOT_CONVERSATION_URL")

    def app_callback(query: str) -> str:
        # Create a CopilotMessage with the query
        copilot_message = CopilotMessage(message=query, user_id="red_team_user")
        
        # Call copilot_chat and get the response
        result = copilot_chat(
            copilot_message,
            copilot_token_secret,
            copilot_conversation_url
        )
        
        # Return the response text
        return result["response"]


    # Define a simple callback function that always returns a fixed response
    # def financial_advisor_callback(query: str) -> str:  # noqa: ARG001
    #     return "I'm a financial advisor assistant. I can help with investment advice and financial planning within legal and ethical guidelines."

    logger.info(f"Initiating Red Teaming Scan...")

    azure_ai_project = AZURE_AI_PROJECT

    red_teaming_display_name = f"Red Teaming-Copilot-{eval_timestamp}"

    red_team_agent = RedTeam(
        azure_ai_project=azure_ai_project, 
        credential=credential,
        risk_categories=[ # optional, defaults to all four risk categories
        RiskCategory.Violence,
        RiskCategory.HateUnfairness,
        RiskCategory.Sexual,
        RiskCategory.SelfHarm
        ], 
        num_objectives=2, # optional, defaults to 10
    )

    # red_team_result = await red_team_agent.scan(target=simple_callback)
    result = await red_team_agent.scan(
        target=app_callback,
        scan_name=red_teaming_display_name,
        attack_strategies=[AttackStrategy.Flip,AttackStrategy.Jailbreak,AttackStrategy.Tense],
        output_path="red_team_output.json",
    )

asyncio.run(run_red_team_scan())
