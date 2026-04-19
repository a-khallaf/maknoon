package commands

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/al-Zamakhshari/maknoon/pkg/crypto"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/spf13/cobra"
)

var (
	chatPassphrase string
	chatSignKey    string
)

const chatAppID = "maknoon.io/ghost-chat/v1"

// ChatCmd returns the cobra command for ghost chat.
func ChatCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "chat [code]",
		Short: "Start a secure, ephemeral Ghost Chat session",
		Long:  `Opens a real-time, end-to-end encrypted P2P chat room. Zero permanent logs.`,
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if JSONOutput {
				return runAgentChat(args)
			}
			return runTuiChat(args)
		},
	}

	cmd.Flags().StringVarP(&chatPassphrase, "passphrase", "s", "", "Shared secret for the chat (optional)")
	cmd.Flags().StringVar(&chatSignKey, "sign-key", "", "Path to your private key for message signing")

	return cmd
}

// --- Agent Mode (JSONL REPL) ---

func runAgentChat(args []string) error {
	sess := crypto.NewChatSession(chatAppID)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var code string
	var err error
	if len(args) > 0 {
		code = args[0]
		err = sess.StartJoin(ctx, code)
	} else {
		code, err = sess.StartHost(ctx)
	}

	if err != nil {
		printErrorJSON(err)
		return err
	}

	printJSON(map[string]interface{}{
		"event": "status",
		"state": "established",
		"code":  code,
	})

	// Input loop
	go func() {
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			var cmd map[string]interface{}
			if err := json.Unmarshal(scanner.Bytes(), &cmd); err != nil {
				continue
			}
			if action, _ := cmd["action"].(string); action == "send" {
				if text, ok := cmd["text"].(string); ok {
					_ = sess.Send(ctx, text)
				}
			} else if action == "quit" {
				sess.Close()
				os.Exit(0)
			}
		}
	}()

	// Event loop
	for ev := range sess.Events {
		printJSON(ev)
	}

	return nil
}

// --- Human Mode (Bubbletea TUI) ---

type chatModel struct {
	sess        *crypto.ChatSession
	viewport    viewport.Model
	textarea    textarea.Model
	messages    []string
	senderStyle lipgloss.Style
	code        string
	connected   bool
}

func (m chatModel) Init() tea.Cmd {
	return textarea.Blink
}

func (m chatModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var (
		tiCmd tea.Cmd
		vpCmd tea.Cmd
	)

	m.textarea, tiCmd = m.textarea.Update(msg)
	m.viewport, vpCmd = m.viewport.Update(msg)

	switch msg := msg.(type) {
	case crypto.ChatEvent:
		switch msg.Type {
		case "status":
			if msg.State == "established" {
				m.code = msg.Text
				m.viewport.SetContent("✅ Connected to Wormhole. Waiting for peer...")
			} else if msg.State == "peer-joined" {
				m.connected = true
				m.viewport.SetContent("⚡ Peer joined the room. Real-time sync active.")
			}
		case "message":
			sender := lipgloss.NewStyle().Foreground(lipgloss.Color("2")).Bold(true).Render("Peer: ")
			m.messages = append(m.messages, sender+msg.Text)
			m.viewport.SetContent(strings.Join(m.messages, "\n"))
			m.viewport.GotoBottom()
		case "error":
			m.messages = append(m.messages, lipgloss.NewStyle().Foreground(lipgloss.Color("9")).Render("Error: ")+msg.Text)
			m.viewport.SetContent(strings.Join(m.messages, "\n"))
		}

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyEsc:
			m.sess.Close()
			return m, tea.Quit
		case tea.KeyEnter:
			content := m.textarea.Value()
			if content == "" {
				return m, nil
			}
			err := m.sess.Send(context.Background(), content)
			if err != nil {
				m.messages = append(m.messages, lipgloss.NewStyle().Foreground(lipgloss.Color("9")).Render("Send Error: ")+err.Error())
			} else {
				m.messages = append(m.messages, m.senderStyle.Render("You: ")+content)
			}
			m.viewport.SetContent(strings.Join(m.messages, "\n"))
			m.textarea.Reset()
			m.viewport.GotoBottom()
			return m, nil
		}

	case tea.WindowSizeMsg:
		m.viewport.Width = msg.Width
		m.textarea.SetWidth(msg.Width)
		m.viewport.Height = msg.Height - m.textarea.Height() - 4
	}

	return m, tea.Batch(tiCmd, vpCmd)
}

func (m chatModel) View() string {
	header := lipgloss.NewStyle().
		Background(lipgloss.Color("5")).
		Foreground(lipgloss.Color("15")).
		Padding(0, 1).
		Bold(true).
		Render(" GHOST CHAT ")

	status := fmt.Sprintf("  Code: %s", m.code)
	if m.connected {
		status += lipgloss.NewStyle().Foreground(lipgloss.Color("2")).Render("  ● ONLINE")
	} else {
		status += lipgloss.NewStyle().Foreground(lipgloss.Color("3")).Render("  ○ CONNECTING")
	}

	return header + status + "\n" +
		strings.Repeat("─", m.viewport.Width) + "\n" +
		m.viewport.View() + "\n" +
		m.textarea.View() + "\n" +
		lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Render("Press Esc to exit • Ephemeral Session")
}

func runTuiChat(args []string) error {
	sess := crypto.NewChatSession(chatAppID)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ta := textarea.New()
	ta.Placeholder = "Type a message..."
	ta.Focus()
	ta.SetHeight(3)

	vp := viewport.New(80, 20)
	vp.SetContent("🕳️ Opening wormhole...")

	m := chatModel{
		sess:        sess,
		textarea:    ta,
		viewport:    vp,
		messages:    []string{},
		senderStyle: lipgloss.NewStyle().Foreground(lipgloss.Color("5")).Bold(true),
		code:        "pending...",
	}

	p := tea.NewProgram(m, tea.WithAltScreen())

	// Async Connection & Event Bridge
	go func() {
		var code string
		var err error
		if len(args) > 0 {
			code = args[0]
			err = sess.StartJoin(ctx, code)
		} else {
			code, err = sess.StartHost(ctx)
		}

		if err != nil {
			p.Send(crypto.ChatEvent{Type: "error", Text: err.Error()})
			return
		}

		p.Send(crypto.ChatEvent{Type: "status", State: "established", Text: code})

		for ev := range sess.Events {
			p.Send(ev)
		}
	}()

	_, err := p.Run()
	return err
}
