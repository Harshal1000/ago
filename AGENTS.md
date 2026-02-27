# AGENTS.md - Coding Guidelines for ago

> Guidelines for AI agents working in the `ago` Go framework repository.

## Build / Lint / Test Commands

There is not any test files, and i don't want you to create test file as well. Its waste of time.

```bash
# Build
go build ./...

# Lint (standard Go)
go vet ./...

# Format code
go fmt ./...

# Download dependencies
go mod tidy
```

## Project Structure

```
/Users/harshalpatil/Documents/Personal/AgentBuilder/
├── ago.go           # Core types (Content, Part, LLM interface)
├── executor.go      # Agent execution loop (Run, RunSSE)
├── tool.go          # Tool interface and ToolResult
├── agent/           # Agent registry and configuration
│   └── agent.go
├── tools/           # Built-in tools
│   ├── agent_tool.go
│   └── function_tool.go
├── llm/             # LLM provider implementations
│   ├── genai.go     # Google GenAI (Gemini) provider
│   └── openai.go    # OpenAI provider
└── cmd/example/     # Example application
    └── main.go
```

## Code Style Guidelines

### Imports
- Use Go standard import grouping: stdlib, external, internal
- Never use dot imports
- Use full import paths: `github.com/Harshal1000/ago`

### Formatting
- Use `gofmt` for all Go files
- Max line length: ~100 characters (soft limit)
- Indent with tabs (Go standard)
- No trailing whitespace

### Naming Conventions
- **Exported types/functions**: PascalCase (e.g., `GenerateConfig`, `NewTextContent`)
- **Unexported types/functions**: camelCase (e.g., `buildToolMap`, `extractFunctionCalls`)
- **Constants**: Use const blocks with type declarations
- **Interfaces**: Noun names (e.g., `LLM`, `Tool`, `AgentConfig`)
- **Acronyms**: Keep uppercase (e.g., `LLM`, `SSE`, `API`)

### Types
- Prefer explicit types over `interface{}` (use `any` alias in Go 1.18+)
- Use pointer receivers for mutating methods
- Use value receivers for read-only methods
- Document exported types with comments starting with the type name

### Error Handling
- Wrap errors with context: `fmt.Errorf("ago: generate: %w", err)`
- Use error prefix `"ago:"` for package-level errors
- Distinguish between infrastructure errors (stop execution) and tool errors (return to model)
- Check context cancellation explicitly: `if err := ctx.Err(); err != nil { return nil, err }`

### Comments
- Start with the name of the thing being documented
- Use complete sentences
- Group related declarations with section comments: `// ----------- Section -----------`

### Code Patterns

**Interface Implementation Check:**
```go
var _ ago.Tool = (*MyTool)(nil)
```

**Context Handling:**
```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()
```

**Error Wrapping:**
```go
return nil, fmt.Errorf("ago: agent %q has no LLM configured", agent.GetName())
```

**Config Copy Pattern:**
```go
copy := *cfg
cfg = &copy
```

## Dependencies
- Minimal external dependencies
- Currently uses: `google.golang.org/genai`, `github.com/openai/openai-go/v3`
- Go version: 1.24.3

## Architecture Notes
- Core types live in root `ago` package
- Agent configuration in `agent/` subpackage
- Tool implementations in `tools/` subpackage
- LLM providers in `llm/` package (flat structure, one file per provider)
- Executor handles the agentic loop (tool calls in parallel)
- To use a provider, import its package for side effects:
  ```go
  import _ "github.com/Harshal1000/ago/llm"
  ```
