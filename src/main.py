import os

import boto3
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage


def get_llm():
    load_dotenv(override=True)
    client = boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        verify=False,
    )
    return ChatBedrockConverse(
        model=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
        client=client,
    )


def test_connection():
    """Quick connectivity test against Bedrock."""
    llm = get_llm()
    try:
        response = llm.invoke([HumanMessage(content="Say hello world")])
        print(f"Bedrock OK: {response.content}")
    except Exception as e:
        print(f"Bedrock connection failed: {e}")


if __name__ == "__main__":
    test_connection()
