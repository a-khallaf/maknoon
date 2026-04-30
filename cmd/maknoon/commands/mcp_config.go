package commands

import (
	"context"
	"fmt"

	"github.com/al-Zamakhshari/maknoon/pkg/crypto"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

func registerConfigTools(s *server.MCPServer, engine crypto.MaknoonEngine) {
	// 1. Tool: config_update
	updateHandler := func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		args := getArgs(request)
		profileID, _ := args["profile_id"].(float64)

		conf := engine.GetConfig()
		if profileID != 0 {
			conf.Performance.DefaultProfile = byte(profileID)
		}

		err := engine.UpdateConfig(&crypto.EngineContext{Context: ctx}, conf)
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}

		return mcp.NewToolResultText(fmt.Sprintf("Configuration updated: DefaultProfile=%d", conf.Performance.DefaultProfile)), nil
	}

	s.AddTool(mcp.NewTool("config_update",
		mcp.WithDescription("Update engine configuration (e.g., change default profile)"),
	), updateHandler)
}
