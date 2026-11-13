"""MCP transport for the synthetic claim projection."""

import argparse
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from .claims import get_synthetic_claim
from .fixtures import load_fixture

TOOL_NAME = "get_synthetic_claim"


class LookupResult(BaseModel):
    status: str
    fixture_reads: int
    claim: dict | None = None


def serve(allowed: bool) -> None:
    server = FastMCP("synthetic-billing-claims")

    @server.tool(name=TOOL_NAME, annotations=ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False,
    ))
    def lookup(claim_id: str) -> LookupResult:
        """Retrieve the allowlisted fields of one synthetic claim."""
        reads = 0

        def read_claims():
            nonlocal reads
            reads += 1
            return load_fixture("claims.json")

        result = get_synthetic_claim(claim_id, allowed=allowed, load_claims=read_claims)
        return LookupResult(**result, fixture_reads=reads)

    server.run(transport="stdio")


async def lookup_claim(claim_id: str, *, server_allowed: bool) -> dict:
    # Only the local launcher sets the server permission. MCP arguments are data.
    args = ["-m", "billing_assistant.mcp_claims"]
    if server_allowed:
        args.append("--allow-lookup")
    env = {key: os.environ[key] for key in ("PATH", "PYTHONPATH") if key in os.environ}
    params = StdioServerParameters(command=sys.executable, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.call_tool(TOOL_NAME, {"claim_id": claim_id})
            if response.isError or not isinstance(response.structuredContent, dict):
                raise RuntimeError("The synthetic claim tool returned an error.")
            return LookupResult.model_validate(response.structuredContent).model_dump(exclude_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-lookup", action="store_true")
    serve(parser.parse_args().allow_lookup)
