# ago

**Production-grade AI agent framework for Go.**

Build agents that think, act, and compose — with the performance and reliability Go is known for.

```
go get github.com/Harshal1000/ago
```

---

## Why ago?

Most AI frameworks are Python-first, abstraction-heavy, and break in production. ago is different.

- **Native Go** — struct literals, interfaces, context propagation. No magic.
- **Real agentic loop** — tool calling, parallel execution, multi-turn reasoning. Not a wrapper around an API call.
- **Agents as tools** — compose agents into hierarchies. An agent can delegate to sub-agents seamlessly.
- **Streaming built-in** — SSE-ready streaming with Go 1.23 iterators. `for range` over your agent's output.
- **Backend agnostic** — swap LLM providers without changing agent code. One import switches everything.
- **Production errors** — two error lanes: recoverable tool errors (sent back to the model) and infrastructure failures (stop the loop). Your agent handles both correctly.

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/Harshal1000/ago"
    "github.com/Harshal1000/ago/agent"
    "github.com/Harshal1000/ago/tools"
    _ "github.com/Harshal1000/ago/llm" // register backends
)

func main() {
    // Define a tool
    search := &tools.FunctionTool{
        ToolName:    "search",
        Description: "Search for information",
        Parameters: &ago.Schema{
            Type: ago.TypeObject,
            Properties: map[string]*ago.Schema{
                "query": {Type: ago.TypeString, Description: "search query"},
            },
            Required: []string{"query"},
        },
        Fn: func(ctx context.Context, args map[string]any) (map[string]any, error) {
            query := args["query"].(string)
            return map[string]any{"results": doSearch(query)}, nil
        },
    }

    // Create an agent
    a := &agent.Agent{
        Name:         "assistant",
        Backend:      agent.BackendGenAI,
        Model:        "gemini-2.0-flash",
        SystemPrompt: "You are a helpful assistant.",
        Tools:        []ago.Tool{search},
    }
    a.InitLLM()

    // Run it
    result, err := ago.Run(context.Background(), a, []*ago.Content{
        ago.NewTextContent(ago.RoleUser, "Find the latest Go release notes"),
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(result.Response.Candidates[0].Content.Parts[0].Text)
}
```

That's it. The agent calls the LLM, decides to use the search tool, executes it, feeds the result back, and returns a synthesized response. All automatic.

## Core Concepts

### Agents

An agent is a configuration: a model, a system prompt, and a set of tools. No inheritance, no base classes — just a struct.

```go
a := &agent.Agent{
    Name:          "researcher",
    Backend:       agent.BackendGenAI,
    Model:         "gemini-2.0-flash",
    SystemPrompt:  "You research topics thoroughly.",
    Tools:         []ago.Tool{searchTool, readTool},
    MaxIterations: 10,
}
```

### Tools

Any Go function becomes a tool. Implement the `Tool` interface or use `FunctionTool` for zero boilerplate.

```go
calculator := &tools.FunctionTool{
    ToolName:    "calculate",
    Description: "Evaluate a math expression",
    Parameters:  &ago.Schema{
        Type: ago.TypeObject,
        Properties: map[string]*ago.Schema{
            "expression": {Type: ago.TypeString},
        },
        Required: []string{"expression"},
    },
    Fn: func(ctx context.Context, args map[string]any) (map[string]any, error) {
        expr := args["expression"].(string)
        result := evaluate(expr)
        return map[string]any{"result": result}, nil
    },
}
```

### Agent Composition

Wrap any agent as a tool. The sub-agent runs its own full loop with its own tools.

```go
// Register a specialist agent
agent.Register(&agent.Agent{
    Name:         "code-reviewer",
    Model:        "gemini-2.0-flash",
    SystemPrompt: "You review code for bugs and security issues.",
    Tools:        []ago.Tool{readFileTool, lintTool},
})

// Use it as a tool in another agent
reviewTool := &tools.AgentTool{
    ToolName:    "code_review",
    Description: "Run a thorough code review",
    AgentName:   "code-reviewer",
}
```

### Streaming

Stream responses with Go iterators. Works through tool calls — the stream pauses during tool execution and resumes when the LLM continues.

```go
for chunk, err := range ago.RunSSE(ctx, agent, contents) {
    if err != nil {
        log.Fatal(err)
    }
    for _, p := range chunk.Candidates[0].Content.Parts {
        fmt.Print(p.Text)
    }
}
```

### SkipSynthesis

When a tool's output IS the answer, skip the extra LLM call. Saves tokens and latency.

```go
apiLookup := &tools.FunctionTool{
    ToolName: "get_weather",
    Fn:       fetchWeather,
    ToolOptions: ago.ToolOptions{SkipSynthesis: true},
    // ...
}
```

## Error Handling

ago distinguishes between two kinds of errors:

| Error Type | What Happens | Example |
|---|---|---|
| **Tool error** | Sent to the model as context. Loop continues. The LLM can retry or work around it. | API rate limit, invalid input, no results found |
| **Infrastructure error** | Loop stops immediately. Returned to caller. | Network down, context cancelled, invalid credentials |

```go
// Tool error — return it in the result, loop continues
return &ago.ToolResult{Error: fmt.Errorf("no results found")}, nil

// Infrastructure error — return as Go error, loop stops
return nil, fmt.Errorf("database connection lost")
```

## Context & Cancellation

`context.Context` flows through the entire execution chain. Set deadlines, propagate cancellation, pass values — it all works exactly as Go developers expect.

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

result, err := ago.Run(ctx, agent, contents)
// Timeout cancels the LLM call, any running tools, and sub-agents
```

## Requirements

- Go 1.23+
- A supported LLM backend API key (e.g. `GOOGLE_API_KEY` for Gemini)

## License

MIT
