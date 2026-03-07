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
	"github.com/Harshal1000/ago/plugins"
	"github.com/Harshal1000/ago/storage"
	"github.com/Harshal1000/ago/tools"
	"github.com/google/uuid"
)

func main() {
	if os.Getenv("GEMINI_API_KEY") == "" {
		log.Fatal("Set GEMINI_API_KEY environment variable")
	}

	svc, err := storage.New(context.Background(), &storage.Config{
		DatabaseURL: os.Getenv("DATABASE_URL"),
		MaxConns:    20,
	})
	if err != nil {
		log.Fatalf("Failed to init storage: %v", err)
	}
	defer svc.Close()

	app := &ago.App{
		Storage:        svc,
		HistoryLimit:   50,
		IncludeHistory: true,
		Hooks:          plugins.LoggingHooks(nil),
	}

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
					return map[string]any{"count": len(strings.Fields(text))}, nil
				},
			},
			&tools.FunctionTool{
				ToolName:    "current_time",
				Description: "Get the current date and time",
				Parameters:  &ago.Schema{Type: ago.TypeObject, Properties: map[string]*ago.Schema{}},
				Fn: func(ctx context.Context, args map[string]any) (map[string]any, error) {
					now := time.Now()
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
			return map[string]any{"result": result}, nil
		},
	}

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

	var sessionID string
	opts := &ago.RunOptions{UserID: uuid.New().String(), Author: mainAgent.Name}

	turns := []string{
		"My name is Harshal. What is 25 * 4?",
		"Now take that result and add 100 to it.",
		"What's my name? Also count the words in: 'The quick brown fox jumps over the lazy dog'",
		"What time is it right now?",
		"Summarize what we've done in this conversation.",
	}

	for _, userMsg := range turns {
		opts.SessionID = sessionID
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)

		result, err := app.Run(ctx, mainAgent, []*ago.Content{
			ago.NewTextContent(ago.RoleUser, userMsg),
		}, opts)
		cancel()

		if err != nil {
			log.Printf("run error: %v", err)
			continue
		}

		if sessionID == "" {
			sessionID = result.SessionID
		}
	}

	opts.SessionID = sessionID
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	for _, err := range app.RunSSE(ctx, mainAgent, []*ago.Content{
		ago.NewTextContent(ago.RoleUser, "Give me the square root of the final number from our conversation."),
	}, opts) {
		if err != nil {
			log.Printf("stream error: %v", err)
			break
		}
	}
}
