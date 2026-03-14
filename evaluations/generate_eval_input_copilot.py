import json
import logging
from pathlib import Path

from appconfig import AppConfigClient
from copilot_client import copilot_chat, CopilotMessage

# Module-level logger and configuration variables
logger = None
copilot_token_secret = None
copilot_token_url = None
copilot_conversation_url = None

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Uncomment to see DEBUG logs:
    # logging.getLogger().setLevel(logging.DEBUG)


def main():
    global logger, copilot_token_secret, copilot_token_url, copilot_conversation_url
    
    setup_logging()
    logger = logging.getLogger("generate_eval_input")

    logger.info("🔧 Loading App Configuration settings")

    # credential = ChainedTokenCredential(ManagedIdentityCredential(), AzureCliCredential())
    cfg = AppConfigClient()
    copilot_token_secret = cfg.get("COPILOT_TOKEN_SECRET")
    # copilot_token_url no longer needed since we skip token generation
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


            copilot_item = CopilotMessage(**copilot_json)
            resp = copilot_chat(
                copilot_item,
                copilot_token_secret,
                copilot_conversation_url
            )
            response_text = resp["response"]
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
