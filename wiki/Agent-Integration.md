# AI Agent Integration

Maknoon is a first-class citizen for AI agents and LLM-based assistants. It follows the **Model Context Protocol (MCP)** and provides native automated discovery and toolkits for agentic workflows.

## 🤖 Agent Handshake
Maknoon automatically switches to JSON mode if:
1.  The `MAKNOON_AGENT_MODE=1` environment variable is set.
2.  The output is not a TTY (piped or redirected).

This allows agents to use Maknoon without being configured with the `--json` flag explicitly.

## 🔌 MCP Server
Maknoon includes a native Go-based MCP server in `integrations/mcp`. 

### Available Tools
*   `inspect_file`: Returns JSON metadata about an encrypted file.
*   `encrypt_file` / `decrypt_file`: Direct file-to-file or file-to-stdout protection.
*   `gen_password` / `gen_passphrase`: Generates high-entropy secrets for the agent to use.
*   `vault_get` / `vault_set`: Manage credentials in the secure bbolt-backed vault.
*   `identity_active`: Automatically lists public keys found on the system.
*   `identity_publish`: Anchors the agent's identity to a global handle (Nostr/DNS).

### Configuration for Claude Desktop
Add the following to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "maknoon": {
      "command": "go",
      "args": ["run", "/path/to/maknoon/integrations/mcp/main.go"],
      "env": {
        "MAKNOON_BINARY": "/path/to/maknoon/maknoon",
        "MAKNOON_AGENT_MODE": "1"
      }
    }
  }
}
```

## 🏗 Self-Describing CLI (Schema)
The most powerful tool for agents is the `maknoon schema` command. It outputs a recursive JSON-Schema of every available command, flag, and argument.

### Usage
```bash
maknoon schema
```
Agents should run this command when first interacting with Maknoon to build an internal map of supported flags (e.g., `--nostr`, `--stealth`, `--compress`) without requiring a manual README lookup.

## 🛠 Automation Variables
*   `MAKNOON_AGENT_MODE`: Set to `1` to enable the Agent Handshake.
*   `MAKNOON_PASSPHRASE`: Non-interactive master key for vault/identity operations.
*   `MAKNOON_PASSWORD`: Non-interactive secret for `vault set` or password generation.
*   `MAKNOON_PRIVATE_KEY`: Default path to the agent's private identity.
*   `MAKNOON_PUBLIC_KEY`: Default path for encryption recipients.
