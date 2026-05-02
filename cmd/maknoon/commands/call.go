package commands

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func CallCmd() *cobra.Command {
	var addr string
	var argsStr string

	cmd := &cobra.Command{
		Use:   "call [tool_name]",
		Short: "Invoke an MCP tool on a running Maknoon agent via a standard SSE client",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			toolName := args[0]
			ctx := context.Background()

			var arguments map[string]any
			if argsStr != "" {
				if err := json.Unmarshal([]byte(argsStr), &arguments); err != nil {
					return fmt.Errorf("invalid JSON arguments: %v", err)
				}
			}

			// Ensure address has protocol and standard SSE path
			baseURL := addr
			if !strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://") {
				baseURL = "http://" + baseURL
			}
			if !strings.HasSuffix(baseURL, "/sse") {
				baseURL = strings.TrimSuffix(baseURL, "/") + "/sse"
			}

			// STANDARD PATH: Formal MCP Client Lifecycle
			if viper.GetBool("trace") {
				fmt.Fprintf(os.Stderr, "TRACE: Initializing standard MCP SSE client for %s\n", baseURL)
			}
			mcpClient, err := client.NewSSEMCPClient(baseURL)
			if err != nil {
				return fmt.Errorf("failed to create MCP client: %v", err)
			}
			defer mcpClient.Close()

			// 1. Start (begins SSE event loop and connects)
			if viper.GetBool("trace") {
				fmt.Fprintf(os.Stderr, "TRACE: Starting SSE event loop...\n")
			}
			if err := mcpClient.Start(ctx); err != nil {
				return fmt.Errorf("failed to start client: %v", err)
			}

			// 2. Initialize (Handshake)
			initReq := mcp.InitializeRequest{
				Params: mcp.InitializeParams{
					ProtocolVersion: mcp.LATEST_PROTOCOL_VERSION,
					ClientInfo: mcp.Implementation{
						Name:    "maknoon-orchestrator",
						Version: "v1.0",
					},
				},
			}
			if viper.GetBool("trace") {
				fmt.Fprintf(os.Stderr, "TRACE: Sending Initialize request...\n")
			}
			if _, err := mcpClient.Initialize(ctx, initReq); err != nil {
				return fmt.Errorf("initialization failed: %v", err)
			}

			// 3. Call Tool
			callReq := mcp.CallToolRequest{
				Params: mcp.CallToolParams{
					Name:      toolName,
					Arguments: arguments,
				},
			}
			if viper.GetBool("trace") {
				fmt.Fprintf(os.Stderr, "TRACE: Calling tool '%s'...\n", toolName)
			}

			result, err := mcpClient.CallTool(ctx, callReq)
			if err != nil {
				return fmt.Errorf("tool execution failed: %v", err)
			}

			// Render result as JSON
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(result)
		},
	}

	cmd.Flags().StringVar(&addr, "addr", "localhost:8080", "Address of the running Maknoon agent")
	cmd.Flags().StringVar(&argsStr, "args", "", "JSON string of tool arguments")
	return cmd
}
