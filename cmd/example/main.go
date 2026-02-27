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
	"github.com/Harshal1000/ago/tools"
)

func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("Set OPENAI_API_KEY environment variable")
	}

	// --- Define tools ---

	calculatorTool := &tools.FunctionTool{
		ToolName:    "calculator",
		Description: "Perform basic math operations. Supports add, subtract, multiply, divide, sqrt, power.",
		Parameters: &ago.Schema{
			Type: ago.TypeObject,
			Properties: map[string]*ago.Schema{
				"operation": {Type: ago.TypeString, Description: "Math operation: add, subtract, multiply, divide, sqrt, power", Enum: []string{"add", "subtract", "multiply", "divide", "sqrt", "power"}},
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

	timeTool := &tools.FunctionTool{
		ToolName:    "current_time",
		Description: "Get the current date and time",
		Parameters:  &ago.Schema{Type: ago.TypeObject, Properties: map[string]*ago.Schema{}},
		Fn: func(ctx context.Context, args map[string]any) (map[string]any, error) {
			now := time.Now()
			fmt.Printf("  [current_time] %s\n", now.Format(time.RFC3339))
			return map[string]any{
				"datetime": now.Format(time.RFC3339),
				"unix":     now.Unix(),
			}, nil
		},
		ToolOptions: ago.ToolOptions{SkipSynthesis: true}, // return directly, no extra LLM call
	}

	wordCountTool := &tools.FunctionTool{
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
			words := len(strings.Fields(text))
			fmt.Printf("  [word_count] %d words\n", words)
			return map[string]any{"count": words}, nil
		},
	}

	// --- Create agent ---

	a := &agent.Agent{
		Name:    "demo-agent",
		Backend: agent.BackendOpenAI,
		Model:   "gpt-4o",
		SystemPrompt: `You are a helpful assistant with access to tools.
Use the calculator for any math. Use current_time when asked about time/date.
Use word_count to count words. Always use tools rather than guessing.`,
		Tools:         []ago.Tool{calculatorTool, timeTool, wordCountTool},
		MaxIterations: 5,
	}

	if err := a.InitLLM(); err != nil {
		log.Fatalf("Failed to init LLM: %v", err)
	}

	// --- Test cases ---

	tests := []struct {
		name  string
		input string
	}{
		{
			name:  "Simple chat (no tools)",
			input: "What is the capital of France?",
		},
		{
			name:  "Single tool call",
			input: "What is 42 * 17?",
		},
		{
			name:  "Multi-step math",
			input: "What is the square root of 144, then multiply that by 5?",
		},
		{
			name:  "SkipSynthesis tool",
			input: "What time is it right now?",
		},
		{
			name:  "Multiple tools in conversation",
			input: "Count the words in 'the quick brown fox jumps over the lazy dog' and also calculate 2 to the power of 10",
		},
	}

	for _, tc := range tests {
		fmt.Printf("\n%s\n%s\n", strings.Repeat("=", 60), tc.name)
		fmt.Printf("User: %s\n", tc.input)
		fmt.Println(strings.Repeat("-", 40))

		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)

		result, err := ago.Run(ctx, a, []*ago.Content{
			ago.NewTextContent(ago.RoleUser, tc.input),
		})
		cancel()

		if err != nil {
			fmt.Printf("ERROR: %v\n", err)
			continue
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

	// --- Streaming example ---

	fmt.Printf("\n%s\nStreaming example\n", strings.Repeat("=", 60))
	fmt.Println("User: Calculate 123 + 456 and tell me the result")
	fmt.Println(strings.Repeat("-", 40))
	fmt.Print("Agent: ")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	for chunk, err := range ago.RunSSE(ctx, a, []*ago.Content{
		ago.NewTextContent(ago.RoleUser, "Calculate 123 + 456 and tell me the result"),
	}) {
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
