"""
Evaluation script using Azure AI Projects SDK 2.0.0b1+
Migrated from 1.0.0b11 to use the new OpenAI-based evals API.
"""

import sys
import logging
import os
import json
import time
from pathlib import Path
from datetime import datetime, UTC
import asyncio
from pprint import pprint

from azure.identity import ChainedTokenCredential, EnvironmentCredential, ManagedIdentityCredential, AzureCliCredential

import azure.ai.projects
print(f"azure-ai-projects version: {azure.ai.projects.__version__}")

# Debug: Verify environment variables are set (don't print secrets!)
print(f"DEBUG: AZURE_CLIENT_ID set: {bool(os.getenv('AZURE_CLIENT_ID'))}")
print(f"DEBUG: AZURE_TENANT_ID set: {bool(os.getenv('AZURE_TENANT_ID'))}")
print(f"DEBUG: AZURE_CLIENT_SECRET set: {bool(os.getenv('AZURE_CLIENT_SECRET'))}")

from azure.ai.projects import AIProjectClient
from azure.core.exceptions import HttpResponseError

# New imports for 2.0.0b1 evaluation API
from openai.types.evals.create_eval_jsonl_run_data_source_param import (
    CreateEvalJSONLRunDataSourceParam,
    SourceFileID,
)
from openai.types.eval_create_params import DataSourceConfigCustom

from appconfig import AppConfigClient
from keyvault import KeyVaultClient

# Red team configuration - calls the app via HTTP endpoint
RED_TEAM_ENABLED = os.getenv("ENABLE_RED_TEAM", "true").lower() == "true"
import requests as http_requests  # For red team HTTP calls

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
DATASET_NAME          = cfg.get("DATASET_NAME", "eval-dataset")
DATASET_VERSION       = datetime.now(UTC).strftime("v%Y%m%d%H%M%S")
INPUT_FILE            = cfg.get(
    "EVAL_INPUT_FILE",
    str(Path(__file__).parent.parent / "dataset" / "eval-input.jsonl")
)
AZURE_AI_PROJECT      = cfg.get("AI_FOUNDRY_PROJECT_ENDPOINT")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cloud_evaluation")

# Validate necessary config
if not (PROJECT_ENDPOINT and MODEL_ENDPOINT and MODEL_DEPLOYMENT_NAME and MODEL_API_KEY):
    logger.error("Missing one or more required settings: PROJECT_ENDPOINT, MODEL_ENDPOINT, CHAT_DEPLOYMENT_NAME, or MODEL_API_KEY")
    sys.exit(1)

# 2) Initialize credential and AIProjectClient
credential = ChainedTokenCredential(EnvironmentCredential(), ManagedIdentityCredential(), AzureCliCredential())

# Debug: Test credential and show which one worked
try:
    token = credential.get_token("https://management.azure.com/.default")
    print(f"DEBUG: Credential obtained token successfully, expires: {token.expires_on}")
except Exception as e:
    print(f"DEBUG: Credential failed to get token: {e}")
    sys.exit(1)

# Use context manager pattern as shown in official samples
with (
    AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential) as project_client,
    project_client.get_openai_client() as client,
):
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

    # 4) Define data source configuration schema
    # The schema defines what fields are in the evaluation dataset
    data_source_config = DataSourceConfigCustom(
        {
            "type": "custom",
            "item_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "response": {"type": "string"},
                    "context": {"type": "string"},
                    "truth": {"type": "string"},  # ground_truth field name from your dataset
                },
                "required": ["query", "response"],
            },
            "include_sample_schema": True,
        }
    )

    # 5) Configure testing criteria (evaluators) - NEW 2.0.0b1 format
    # Format: list of dicts with type, name, evaluator_name, initialization_parameters, data_mapping
    testing_criteria = [
        {
            "type": "azure_ai_evaluator",
            "name": "completeness",
            "evaluator_name": "builtin.response_completeness",
            "initialization_parameters": {
                "deployment_name": MODEL_DEPLOYMENT_NAME,
                "threshold": 4,
            },
            "data_mapping": {
                "response": "{{item.response}}",
                "ground_truth": "{{item.truth}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "relevance",
            "evaluator_name": "builtin.relevance",
            "initialization_parameters": {
                "deployment_name": MODEL_DEPLOYMENT_NAME,
            },
            "data_mapping": {
                "response": "{{item.response}}",
                "query": "{{item.query}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "retrieval",
            "evaluator_name": "builtin.retrieval",
            "initialization_parameters": {
                "deployment_name": MODEL_DEPLOYMENT_NAME,
            },
            "data_mapping": {
                "query": "{{item.query}}",
                "context": "{{item.context}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "coherence",
            "evaluator_name": "builtin.coherence",
            "initialization_parameters": {
                "deployment_name": MODEL_DEPLOYMENT_NAME,
            },
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "similarity",
            "evaluator_name": "builtin.similarity",
            "initialization_parameters": {
                "deployment_name": MODEL_DEPLOYMENT_NAME,
            },
            "data_mapping": {
                "response": "{{item.response}}",
                "ground_truth": "{{item.truth}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "groundedness",
            "evaluator_name": "builtin.groundedness",
            "initialization_parameters": {
                "deployment_name": MODEL_DEPLOYMENT_NAME,
            },
            "data_mapping": {
                "query": "{{item.query}}",
                "context": "{{item.context}}",
                "response": "{{item.response}}",
            },
        },
        # Safety evaluators (Content Safety)
        {
            "type": "azure_ai_evaluator",
            "name": "violence",
            "evaluator_name": "builtin.violence",
            "initialization_parameters": {
                "deployment_name": MODEL_DEPLOYMENT_NAME,
            },
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "sexual",
            "evaluator_name": "builtin.sexual",
            "initialization_parameters": {
                "deployment_name": MODEL_DEPLOYMENT_NAME,
            },
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "self_harm",
            "evaluator_name": "builtin.self_harm",
            "initialization_parameters": {
                "deployment_name": MODEL_DEPLOYMENT_NAME,
            },
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "hate_unfairness",
            "evaluator_name": "builtin.hate_unfairness",
            "initialization_parameters": {
                "deployment_name": MODEL_DEPLOYMENT_NAME,
            },
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            },
        },
    ]

    # 6) Create evaluation definition
    commit_id = os.getenv("COMMIT_ID") or (sys.argv[1] if len(sys.argv) > 1 else "unknown")
    eval_timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    eval_name = f"Auto evaluation-{commit_id}-{eval_timestamp}"

    logger.info(f"Creating evaluation definition: {eval_name}")
    try:
        eval_object = client.evals.create(
            name=eval_name,
            data_source_config=data_source_config,
            testing_criteria=testing_criteria,  # type: ignore
        )
        logger.info(f"Evaluation definition created (id: {eval_object.id})")
    except Exception as e:
        logger.error(f"Evaluation definition creation failed: {e}")
        sys.exit(1)

    # 7) Create evaluation run with the uploaded dataset
    run_name = f"eval-run-{commit_id}-{eval_timestamp}"
    logger.info(f"Creating evaluation run: {run_name}")
    try:
        eval_run = client.evals.runs.create(
            eval_id=eval_object.id,
            name=run_name,
            metadata={
                "commit_id": commit_id,
                "timestamp": eval_timestamp,
                "description": "Pre-deployment RAG evaluation",
            },
            data_source=CreateEvalJSONLRunDataSourceParam(
                type="jsonl",
                source=SourceFileID(
                    type="file_id",
                    id=dataset.id if dataset.id else "",
                ),
            ),
        )
        logger.info(f"Evaluation run created (id: {eval_run.id})")
    except Exception as e:
        logger.error(f"Evaluation run creation failed: {e}")
        sys.exit(1)

    # 8) Wait for evaluation to complete and get results
    logger.info("Waiting for evaluation run to complete...")
    while True:
        run = client.evals.runs.retrieve(run_id=eval_run.id, eval_id=eval_object.id)
        if run.status == "completed":
            logger.info(f"Evaluation completed successfully!")
            output_items = list(client.evals.runs.output_items.list(run_id=run.id, eval_id=eval_object.id))
            
            # Save results to file
            results_path = Path(__file__).parent / "evaluation-results.json"
            with open(results_path, "w") as f:
                json.dump([item.model_dump() if hasattr(item, 'model_dump') else item for item in output_items], f, indent=2, default=str)
            logger.info(f"Results saved to: {results_path}")
            
            # Print report URL
            if run.report_url:
                logger.info(f"Evaluation report URL: \033[94m\033[4m{run.report_url}\033[0m")
            break
        elif run.status == "failed":
            logger.error(f"Evaluation run failed!")
            pprint(run)
            sys.exit(1)
        else:
            logger.info(f"Status: {run.status} - waiting...")
            time.sleep(10)

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Evaluation ID: {eval_object.id}")
    logger.info(f"Run ID: {eval_run.id}")
    logger.info(f"Status: {run.status}")
    if run.report_url:
        logger.info(f"Report: {run.report_url}")

# =============================================================================
# RED TEAM SCANNING
# =============================================================================
# Uses HTTP requests to call the running app endpoint (avoids SDK import conflicts)
# The app must be running and accessible at APP_ENDPOINT

if RED_TEAM_ENABLED:
    from azure.ai.evaluation.red_team import RedTeam, RiskCategory, AttackStrategy
    
    # Get app endpoint from config or environment
    APP_ENDPOINT = os.getenv("APP_ENDPOINT") or cfg.get("APP_ENDPOINT") or "http://localhost:8000"
    APP_API_KEY = os.getenv("APP_API_KEY") or cfg.get("APP_API_KEY") or "sample"
    
    async def run_red_team_scan():
        logger.info(f"### Red Teaming Scan Starting ###")
        logger.info(f"Target endpoint: {APP_ENDPOINT}")

        def app_callback(query: str) -> str:
            """Call the app via HTTP POST to /orchestrator endpoint."""
            try:
                resp = http_requests.post(
                    f"{APP_ENDPOINT}/orchestrator",
                    json={"ask": query, "conversation_id": None},
                    headers={"X-API-KEY": APP_API_KEY},
                    timeout=120
                )
                resp.raise_for_status()
                return resp.text
            except http_requests.exceptions.RequestException as e:
                logger.error(f"Red team callback failed: {e}")
                return f"Error: {e}"

        logger.info(f"Initiating Red Teaming Scan...")

        commit_id = os.getenv("COMMIT_ID") or "unknown"
        eval_timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        red_teaming_display_name = f"Red Teaming-{commit_id}-{eval_timestamp}"

        red_team_agent = RedTeam(
            azure_ai_project=AZURE_AI_PROJECT, 
            credential=credential,
            risk_categories=[
                RiskCategory.Violence,
                RiskCategory.HateUnfairness,
                RiskCategory.Sexual,
                RiskCategory.SelfHarm
            ], 
            num_objectives=2,
        )

        result = await red_team_agent.scan(
            target=app_callback,
            scan_name=red_teaming_display_name,
            attack_strategies=[AttackStrategy.Flip, AttackStrategy.Jailbreak, AttackStrategy.Tense],
            output_path="red_team_output.json",
        )
        
        logger.info(f"Red Team scan completed. Results saved to red_team_output.json")

    asyncio.run(run_red_team_scan())
else:
    logger.info("Red team scan disabled. Set ENABLE_RED_TEAM=true to enable.")
