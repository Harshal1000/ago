package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
	"time"

	"github.com/Harshal1000/ago"
	"github.com/Harshal1000/ago/agent"
	_ "github.com/Harshal1000/ago/llm"
	"github.com/Harshal1000/ago/storage"
	"github.com/Harshal1000/ago/tools"
)

func main() {
	if os.Getenv("GEMINI_API_KEY") == "" {
		log.Fatal("Set GEMINI_API_KEY environment variable")
	}

	// =====================================================================
	// Initialize storage (PostgreSQL or in-memory)
	// =====================================================================
	svc, err := storage.New(context.Background(), &storage.Config{
		DatabaseURL: os.Getenv("DATABASE_URL"),
		MaxConns:    20,
	})
	if err != nil {
		log.Fatalf("Failed to init storage: %v", err)
	}
	defer svc.Close()

	if os.Getenv("DATABASE_URL") != "" {
		fmt.Println("Using PostgreSQL storage")
	} else {
		fmt.Println("Using in-memory storage (set DATABASE_URL for PostgreSQL)")
	}

	// =====================================================================
	// App — shared infrastructure for all agents
	// =====================================================================
	app := &ago.App{
		Storage:        svc,
		HistoryLimit:   50, // load at most 50 recent events per turn
		IncludeHistory: true,
	}

	// =====================================================================
	// Sub-agent — cheaper model (gemini-2.0-flash-lite) for simple tasks
	// =====================================================================
	helperAgent := &agent.Agent{
		Name:    "helper-agent",
		Backend: agent.BackendGenAI,
		Model:   "gemini-2.0-flash-lite",
		SystemPrompt: `You are a lightweight helper. You handle simple tasks efficiently:
- Word counting: count words in text precisely
- Current time: report the exact current time
Always give short, direct answers.`,
		Tools: []ago.Tool{
			&tools.FunctionTool{
				ToolName:    "word_count",
				Description: "Count words in a given text",
				Parameters: &ago.Schema{
					Type: ago.TypeObject,
					Properties: map[string]*ago.Schema{
						"text": {Type: ago.TypeString, Description: "Text to count words in"},
					},
					Required: []string{"text"},
				},
				Fn: func(ctx context.Context, args map[string]any) (map[string]any, error) {
					text, _ := args["text"].(string)
					count := len(strings.Fields(text))
					fmt.Printf("    [word_count] %d words\n", count)
					return map[string]any{"count": count}, nil
				},
			},
			&tools.FunctionTool{
				ToolName:    "current_time",
				Description: "Get the current date and time",
				Parameters:  &ago.Schema{Type: ago.TypeObject, Properties: map[string]*ago.Schema{}},
				Fn: func(ctx context.Context, args map[string]any) (map[string]any, error) {
					now := time.Now()
					fmt.Printf("    [current_time] %s\n", now.Format(time.RFC3339))
					return map[string]any{"datetime": now.Format(time.RFC3339), "unix": now.Unix()}, nil
				},
				ToolOptions: ago.ToolOptions{SkipSynthesis: true},
			},
		},
		Config:        &ago.GenerateConfig{MaxOutputTokens: 500},
		MaxIterations: 3,
	}
	if err := helperAgent.InitLLM(); err != nil {
		log.Fatalf("Failed to init helper agent LLM: %v", err)
	}

	// =====================================================================
	// Main agent — full model (gemini-2.0-flash) with calculator + sub-agent
	// =====================================================================
	calculatorTool := &tools.FunctionTool{
		ToolName:    "calculator",
		Description: "Perform basic math operations: add, subtract, multiply, divide, sqrt, power.",
		Parameters: &ago.Schema{
			Type: ago.TypeObject,
			Properties: map[string]*ago.Schema{
				"operation": {Type: ago.TypeString, Description: "Math operation", Enum: []string{"add", "subtract", "multiply", "divide", "sqrt", "power"}},
				"a":         {Type: ago.TypeNumber, Description: "First number"},
				"b":         {Type: ago.TypeNumber, Description: "Second number (not needed for sqrt)"},
			},
			Required: []string{"operation", "a"},
		},
		Fn: func(ctx context.Context, args map[string]any) (map[string]any, error) {
			op, _ := args["operation"].(string)
			a, _ := args["a"].(float64)
			b, _ := args["b"].(float64)
			var result float64
			switch op {
			case "add":
				result = a + b
			case "subtract":
				result = a - b
			case "multiply":
				result = a * b
			case "divide":
				if b == 0 {
					return nil, fmt.Errorf("division by zero")
				}
				result = a / b
			case "sqrt":
				if a < 0 {
					return nil, fmt.Errorf("cannot take sqrt of negative number")
				}
				result = math.Sqrt(a)
			case "power":
				result = math.Pow(a, b)
			default:
				return nil, fmt.Errorf("unknown operation: %s", op)
			}
			fmt.Printf("  [calculator] %s(%g, %g) = %g\n", op, a, b, result)
			return map[string]any{"result": result}, nil
		},
	}

	// Wrap the helper agent as a tool — main agent delegates cheap tasks here.
	// Sub-agent runs ephemerally: no storage, no history, current turn only.
	helperTool := &tools.AgentTool{
		ToolName:    "helper",
		Description: "Delegate simple tasks (word counting, current time) to a lightweight helper agent. Cost-effective for non-math queries.",
		Agent:       helperAgent,
	}

	mainAgent := &agent.Agent{
		Name:    "main-agent",
		Backend: agent.BackendGenAI,
		Model:   "gemini-2.0-flash",
		SystemPrompt: `You are a helpful assistant with access to tools.
Use the calculator for any math operations.
Delegate word counting and time queries to the helper agent — it is cheaper and faster for those tasks.
Always use tools rather than guessing.`,
		Tools: []ago.Tool{calculatorTool, helperTool},
		Config: &ago.GenerateConfig{
			MaxOutputTokens: 1000,
			Temperature:     &[]float64{0.7}[0],
		},
		MaxIterations: 5,
	}
	if err := mainAgent.InitLLM(); err != nil {
		log.Fatalf("Failed to init main agent LLM: %v", err)
	}

	// =====================================================================
	// Multi-turn conversation — same session, model remembers prior turns
	// =====================================================================
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("MULTI-TURN CONVERSATION")
	fmt.Println(strings.Repeat("=", 60))

	var sessionID string
	opts := &ago.RunOptions{UserID: "demo-user", Author: mainAgent.Name}

	turns := []string{
		"My name is Harshal. What is 25 * 4?",
		"Now take that result and add 100 to it.",
		"What's my name? Also count the words in: 'The quick brown fox jumps over the lazy dog'",
		"What time is it right now?",
		"Summarize what we've done in this conversation.",
	}

	for i, userMsg := range turns {
		fmt.Printf("\n--- Turn %d ---\n", i+1)
		fmt.Printf("User: %s\n", userMsg)

		opts.SessionID = sessionID
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)

		result, err := app.Run(ctx, mainAgent, []*ago.Content{
			ago.NewTextContent(ago.RoleUser, userMsg),
		}, opts)
		cancel()

		if err != nil {
			fmt.Printf("ERROR: %v\n", err)
			continue
		}

		if sessionID == "" {
			sessionID = result.SessionID
			fmt.Printf("Session created: %s\n", sessionID)
		}

		if len(result.Response.Candidates) > 0 && result.Response.Candidates[0].Content != nil {
			for _, p := range result.Response.Candidates[0].Content.Parts {
				if p.Text != "" {
					fmt.Printf("Agent: %s\n", p.Text)
				}
			}
		}
		fmt.Printf("History turns: %d\n", len(result.History))
	}

	// =====================================================================
	// Verify stored events
	// =====================================================================
	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Println("STORED EVENTS FOR SESSION")
	fmt.Println(strings.Repeat("=", 60))

	events, err := svc.GetEvents(context.Background(), sessionID)
	if err != nil {
		fmt.Printf("ERROR reading events: %v\n", err)
	} else {
		for i, ev := range events {
			preview := string(ev.Content)
			if len(preview) > 100 {
				preview = preview[:100] + "..."
			}
			fmt.Printf("  %2d. [msg:%s] %s\n", i+1, ev.MessageID[:8], preview)
		}
		fmt.Printf("Total events: %d\n", len(events))
	}

	// =====================================================================
	// Streaming turn (same session)
	// =====================================================================
	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Println("STREAMING TURN (same session)")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("User: Give me the square root of the final number from our conversation.")
	fmt.Print("Agent: ")

	opts.SessionID = sessionID
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	for chunk, err := range app.RunSSE(ctx, mainAgent, []*ago.Content{
		ago.NewTextContent(ago.RoleUser, "Give me the square root of the final number from our conversation."),
	}, opts) {
		if err != nil {
			fmt.Printf("\nSTREAM ERROR: %v\n", err)
			break
		}
		if len(chunk.Candidates) > 0 && chunk.Candidates[0].Content != nil {
			for _, p := range chunk.Candidates[0].Content.Parts {
				if p.Text != "" {
					fmt.Print(p.Text)
				}
			}
		}
	}
	fmt.Println()
}
