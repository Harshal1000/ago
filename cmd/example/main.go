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
		Name:           "ago-example",
		Storage:        svc,
		HistoryLimit:   50,
		IncludeHistory: true,
		Hooks:          plugins.LoggingHooks(nil),
	}

	// ---------------------------------------------------------------------------
	// Build shared tools
	// ---------------------------------------------------------------------------

	calculatorTool := &tools.FunctionTool{
		ToolName:    "calculator",
		Description: "Perform basic math: add, subtract, multiply, divide, sqrt, power.",
		Parameters: &ago.Schema{
			Type: ago.TypeObject,
			Properties: map[string]*ago.Schema{
				"operation": {Type: ago.TypeString, Enum: []string{"add", "subtract", "multiply", "divide", "sqrt", "power"}},
				"a":         {Type: ago.TypeNumber},
				"b":         {Type: ago.TypeNumber},
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

	wordCountTool := &tools.FunctionTool{
		ToolName:    "word_count",
		Description: "Count words in a given text",
		Parameters: &ago.Schema{
			Type: ago.TypeObject,
			Properties: map[string]*ago.Schema{
				"text": {Type: ago.TypeString},
			},
			Required: []string{"text"},
		},
		Fn: func(ctx context.Context, args map[string]any) (map[string]any, error) {
			text, _ := args["text"].(string)
			return map[string]any{"count": len(strings.Fields(text))}, nil
		},
	}

	currentTimeTool := &tools.FunctionTool{
		ToolName:    "current_time",
		Description: "Get the current date and time",
		Parameters:  &ago.Schema{Type: ago.TypeObject, Properties: map[string]*ago.Schema{}},
		Fn: func(ctx context.Context, args map[string]any) (map[string]any, error) {
			now := time.Now()
			return map[string]any{"datetime": now.Format(time.RFC3339)}, nil
		},
		ToolOptions: ago.ToolOptions{SkipSynthesis: true},
	}

	// ---------------------------------------------------------------------------
	// Build agents
	// ---------------------------------------------------------------------------

	mustInit := func(a *agent.Agent) *agent.Agent {
		if err := a.InitLLM(); err != nil {
			log.Fatalf("Failed to init LLM for %s: %v", a.Name, err)
		}
		return a
	}

	plannerAgent := mustInit(&agent.Agent{
		Name:         "planner",
		Backend:      agent.BackendGenAI,
		Model:        "gemini-2.5-flash-lite",
		SystemPrompt: "You are a task planner. Given a goal, produce a concise step-by-step plan. Be brief.",
		Config:       &ago.GenerateConfig{MaxOutputTokens: 300},
	})

	writerAgent := mustInit(&agent.Agent{
		Name:         "writer",
		Backend:      agent.BackendGenAI,
		Model:        "gemini-2.5-flash-lite",
		SystemPrompt: "You are a writer. Given a plan, expand it into clear, readable prose. Be concise.",
		Config:       &ago.GenerateConfig{MaxOutputTokens: 500},
	})

	researchAgent1 := mustInit(&agent.Agent{
		Name:         "researcher-1",
		Backend:      agent.BackendGenAI,
		Model:        "gemini-2.5-flash-lite",
		SystemPrompt: "You are a research specialist focused on technical aspects. Give a 2-3 sentence summary.",
		Config:       &ago.GenerateConfig{MaxOutputTokens: 200},
	})

	researchAgent2 := mustInit(&agent.Agent{
		Name:         "researcher-2",
		Backend:      agent.BackendGenAI,
		Model:        "gemini-2.5-flash-lite",
		SystemPrompt: "You are a research specialist focused on business impact. Give a 2-3 sentence summary.",
		Config:       &ago.GenerateConfig{MaxOutputTokens: 200},
	})

	synthAgent := mustInit(&agent.Agent{
		Name:         "synthesizer",
		Backend:      agent.BackendGenAI,
		Model:        "gemini-2.5-flash-lite",
		SystemPrompt: "You are a synthesizer. Combine research findings from multiple sources into one cohesive summary.",
		Config:       &ago.GenerateConfig{MaxOutputTokens: 400},
	})

	criticAgent := mustInit(&agent.Agent{
		Name:         "critic",
		Backend:      agent.BackendGenAI,
		Model:        "gemini-2.5-flash-lite",
		SystemPrompt: `You are a critic. Review the text and respond with either "APPROVED" if it is good enough, or suggest one specific improvement.`,
		Config:       &ago.GenerateConfig{MaxOutputTokens: 150},
	})

	refinerAgent := mustInit(&agent.Agent{
		Name:         "refiner",
		Backend:      agent.BackendGenAI,
		Model:        "gemini-2.5-flash-lite",
		SystemPrompt: "You are a refiner. Apply the critic's suggestion to improve the text. Output only the improved text.",
		Config:       &ago.GenerateConfig{MaxOutputTokens: 400},
	})

	helperAgent := mustInit(&agent.Agent{
		Name:          "helper",
		Backend:       agent.BackendGenAI,
		Model:         "gemini-2.5-flash-lite",
		SystemPrompt:  "You are a lightweight helper for word counting and time queries. Always use tools, never guess.",
		Tools:         []ago.Tool{wordCountTool, currentTimeTool},
		Config:        &ago.GenerateConfig{MaxOutputTokens: 200},
		MaxIterations: 3,
	})

	coordinatorAgent := mustInit(&agent.Agent{
		Name:    "coordinator",
		Backend: agent.BackendGenAI,
		Model:   "gemini-2.5-flash",
		SystemPrompt: `You are a coordinator. Delegate tasks to the right specialist agents using tool calls.
Use the calculator for math, the helper for word/time queries.
Synthesize results into a final answer.`,
		Tools:         []ago.Tool{calculatorTool},
		Config:        &ago.GenerateConfig{MaxOutputTokens: 600},
		MaxIterations: 5,
	})

	userID := uuid.New().String()
	timeout := 30 * time.Second

	// ---------------------------------------------------------------------------
	// 1. Single agent (app.Runner field)
	// ---------------------------------------------------------------------------
	fmt.Println("\n=== 1. Single Agent ===")
	singleApp := &ago.App{
		Name:           "ago-example",
		Storage:        svc,
		HistoryLimit:   20,
		IncludeHistory: false,
		Hooks:          plugins.LoggingHooks(nil),
		Runner:         plannerAgent,
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	result, err := singleApp.Run(ctx, nil, []*ago.Content{
		ago.NewTextContent(ago.RoleUser, "Plan how to learn Go in 30 days."),
	}, &ago.RunOptions{UserID: userID})
	cancel()
	if err != nil {
		log.Printf("single agent error: %v", err)
	} else {
		fmt.Printf("Planner: %s\n", firstLine(ago.ExtractText(result)))
	}

	// ---------------------------------------------------------------------------
	// 2. Sequential: planner → writer
	// ---------------------------------------------------------------------------
	fmt.Println("\n=== 2. Sequential (planner → writer) ===")
	seqApp := &ago.App{
		Name:           "ago-example",
		Storage:        svc,
		HistoryLimit:   20,
		IncludeHistory: false,
		Hooks:          plugins.LoggingHooks(nil),
		Runner:         ago.Sequential(plannerAgent, writerAgent),
	}
	ctx, cancel = context.WithTimeout(context.Background(), timeout)
	result, err = seqApp.Run(ctx, nil, []*ago.Content{
		ago.NewTextContent(ago.RoleUser, "Write a short article about the benefits of Go for backend development."),
	}, &ago.RunOptions{UserID: userID})
	cancel()
	if err != nil {
		log.Printf("sequential error: %v", err)
	} else {
		fmt.Printf("Final article (first line): %s\n", firstLine(ago.ExtractText(result)))
	}

	// ---------------------------------------------------------------------------
	// 3. Parallel: two researchers → synthesizer
	// ---------------------------------------------------------------------------
	fmt.Println("\n=== 3. Parallel (researcher-1 + researcher-2 → synthesizer) ===")
	parallelApp := &ago.App{
		Name:           "ago-example",
		Storage:        svc,
		HistoryLimit:   20,
		IncludeHistory: false,
		Hooks:          plugins.LoggingHooks(nil),
		Runner:         ago.Parallel(researchAgent1, researchAgent2).Aggregate(synthAgent),
	}
	ctx, cancel = context.WithTimeout(context.Background(), timeout)
	result, err = parallelApp.Run(ctx, nil, []*ago.Content{
		ago.NewTextContent(ago.RoleUser, "What is the impact of AI on software development?"),
	}, &ago.RunOptions{UserID: userID})
	cancel()
	if err != nil {
		log.Printf("parallel error: %v", err)
	} else {
		fmt.Printf("Synthesis (first line): %s\n", firstLine(ago.ExtractText(result)))
	}

	// ---------------------------------------------------------------------------
	// 4. Loop: critic + refiner, max 3 iterations or until approved
	// ---------------------------------------------------------------------------
	fmt.Println("\n=== 4. Loop (critic + refiner, max 3, until APPROVED) ===")
	loopApp := &ago.App{
		Name:           "ago-example",
		Storage:        svc,
		HistoryLimit:   20,
		IncludeHistory: false,
		Hooks:          plugins.LoggingHooks(nil),
		Runner: ago.Loop(criticAgent, refinerAgent).
			Max(3).
			Until(func(output string) bool { return strings.Contains(output, "APPROVED") }),
	}
	ctx, cancel = context.WithTimeout(context.Background(), timeout)
	result, err = loopApp.Run(ctx, nil, []*ago.Content{
		ago.NewTextContent(ago.RoleUser, "Go is a great programming language."),
	}, &ago.RunOptions{UserID: userID})
	cancel()
	if err != nil {
		log.Printf("loop error: %v", err)
	} else {
		fmt.Printf("Final text (first line): %s\n", firstLine(ago.ExtractText(result)))
	}

	// ---------------------------------------------------------------------------
	// 5. Orchestrate: coordinator LLM dispatches workers via tool calls
	// ---------------------------------------------------------------------------
	fmt.Println("\n=== 5. Orchestrate (coordinator with calculator + helper workers) ===")
	orchApp := &ago.App{
		Name:           "ago-example",
		Storage:        svc,
		HistoryLimit:   20,
		IncludeHistory: false,
		Hooks:          plugins.LoggingHooks(nil),
		Runner:         ago.Orchestrate(coordinatorAgent, helperAgent),
	}
	ctx, cancel = context.WithTimeout(context.Background(), timeout)
	result, err = orchApp.Run(ctx, nil, []*ago.Content{
		ago.NewTextContent(ago.RoleUser, "What is 144 * 12? Also, what time is it now?"),
	}, &ago.RunOptions{UserID: userID})
	cancel()
	if err != nil {
		log.Printf("orchestrate error: %v", err)
	} else {
		fmt.Printf("Coordinator answer (first line): %s\n", firstLine(ago.ExtractText(result)))
	}

	// ---------------------------------------------------------------------------
	// 6. Nested: Sequential(Parallel(...).Aggregate(synth), writer)
	// ---------------------------------------------------------------------------
	fmt.Println("\n=== 6. Nested: Sequential(Parallel(r1,r2).Aggregate(synth), writer) ===")
	nestedApp := &ago.App{
		Name:           "ago-example",
		Storage:        svc,
		HistoryLimit:   20,
		IncludeHistory: false,
		Hooks:          plugins.LoggingHooks(nil),
		Runner: ago.Sequential(
			ago.Parallel(researchAgent1, researchAgent2).Aggregate(synthAgent),
			writerAgent,
		),
	}
	ctx, cancel = context.WithTimeout(context.Background(), timeout)
	result, err = nestedApp.Run(ctx, nil, []*ago.Content{
		ago.NewTextContent(ago.RoleUser, "Explain the future of cloud computing."),
	}, &ago.RunOptions{UserID: userID})
	cancel()
	if err != nil {
		log.Printf("nested error: %v", err)
	} else {
		fmt.Printf("Final article (first line): %s\n", firstLine(ago.ExtractText(result)))
	}

	// ---------------------------------------------------------------------------
	// 7. RunSSE on a single agent (streaming path)
	// ---------------------------------------------------------------------------
	fmt.Println("\n=== 7. RunSSE (single agent streaming) ===")
	ctx, cancel = context.WithTimeout(context.Background(), timeout)
	defer cancel()
	opts := &ago.RunOptions{UserID: userID}
	for chunk, err := range app.RunSSE(ctx, plannerAgent, []*ago.Content{
		ago.NewTextContent(ago.RoleUser, "Give me a 3-step plan to learn Docker."),
	}, opts) {
		if err != nil {
			log.Printf("stream error: %v", err)
			break
		}
		if chunk != nil && chunk.Complete {
			fmt.Printf("Stream complete.\n")
		}
	}
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	if len(s) > 120 {
		return s[:120] + "..."
	}
	return s
}
