import os

import boto3
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage


def get_llm():
    """Build the Bedrock client using ONLY the credentials from the environment/.env.

    We pass the keys to boto3 explicitly and fail fast if they are missing, so the
    client can never silently fall back to ambient ~/.aws credentials (which may be
    the wrong account).
    """
    load_dotenv(override=True)
    region = os.getenv("AWS_REGION")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not (region and access_key and secret_key):
        raise RuntimeError(
            "Missing AWS config. Set AWS_REGION, AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY in your .env. Refusing to use ambient "
            "~/.aws credentials to avoid hitting the wrong account."
        )
    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),  # optional, for temp creds
        verify=False,
    )
    return ChatBedrockConverse(
        model=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
        client=client,
    )


def test_connection():
    """Quick connectivity test against Bedrock."""
    try:
        llm = get_llm()
        response = llm.invoke([HumanMessage(content="Say hello world")])
        print(f"Bedrock OK: {response.content}")
    except Exception as e:
        print(f"Bedrock connection failed: {e}")


if __name__ == "__main__":
    test_connection()
