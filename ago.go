// Package ago is a Go framework for building AI agents with tool-calling capabilities.
//
// ago provides a backend-agnostic abstraction over LLM providers (OpenAI, Google GenAI, etc.)
// and an agentic execution loop that handles multi-turn tool calling automatically.
//
// # Architecture Overview
//
//   - Core types (this package): Content, Part, Tool, Schema, and the LLM interface.
//     These are the building blocks shared by all backends and the executor.
//
//   - App (ago.App): The primary entry point. Holds app-level infrastructure (storage,
//     history policy) and exposes Run and RunSSE.
//
//   - Agent (ago/agent): A configuration struct that bundles a backend, model, system prompt,
//     tools, and generation config into a single reusable unit. Implements Runner and Streamer.
//
//   - LLM backends (ago/llm): Implementations of the LLM interface for specific providers.
//     Import ago/llm with a blank import to auto-register all backends via init().
//
// # Quick Start
//
//  1. Create an App with optional storage: app := &ago.App{Storage: svc, IncludeHistory: true}
//  2. Create an agent.Agent with a backend, model, system prompt, and tools.
//  3. Call app.Run(ctx, nil, messages, opts) or app.RunSSE for streaming.
package ago

import (
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Roles
// ---------------------------------------------------------------------------

// Role identifies the sender of a message in a conversation.
type Role string

const (
	RoleUser   Role = "user"
	RoleModel  Role = "model"
	RoleSystem Role = "system"
	RoleTool   Role = "tool"
)

// ---------------------------------------------------------------------------
// Content & Parts
// ---------------------------------------------------------------------------

// Content is a single turn in a conversation: a role plus one or more parts.
type Content struct {
	Role  Role    `json:"role,omitempty"`
	Parts []*Part `json:"parts,omitempty"`
}

// Part is one segment of a Content message. Set exactly one field.
type Part struct {
	Text                string               `json:"text,omitempty"`
	FunctionCall        *FunctionCall        `json:"functionCall,omitempty"`
	FunctionResponse    *FunctionResponse    `json:"functionResponse,omitempty"`
	InlineData          *Blob                `json:"inlineData,omitempty"`
	FileData            *FileData            `json:"fileData,omitempty"`
	ExecutableCode      *ExecutableCode      `json:"executableCode,omitempty"`
	CodeExecutionResult *CodeExecutionResult `json:"codeExecutionResult,omitempty"`
	VideoMetadata       *VideoMetadata       `json:"videoMetadata,omitempty"`
	Thought             bool                 `json:"thought,omitempty"`
	ThoughtSignature    []byte               `json:"thoughtSignature,omitempty"`
}

// FunctionCall represents the model requesting a tool/function invocation.
type FunctionCall struct {
	ID   string         `json:"id,omitempty"`
	Name string         `json:"name,omitempty"`
	Args map[string]any `json:"args,omitempty"`
}

// FunctionResponse is the result of a tool invocation, sent back to the model.
type FunctionResponse struct {
	ID       string         `json:"id,omitempty"`
	Name     string         `json:"name,omitempty"`
	Response map[string]any `json:"response,omitempty"`
}

// Blob holds inline binary data with a MIME type.
type Blob struct {
	MIMEType string `json:"mimeType,omitempty"`
	Data     []byte `json:"data,omitempty"`
}

// FileData references external data by URI.
type FileData struct {
	MIMEType    string `json:"mimeType,omitempty"`
	FileURI     string `json:"fileUri,omitempty"`
	DisplayName string `json:"displayName,omitempty"`
}

// ExecutableCode is code produced by the model for execution.
type ExecutableCode struct {
	Code     string       `json:"code,omitempty"`
	Language CodeLanguage `json:"language,omitempty"`
}

// CodeLanguage identifies the programming language of executable code.
type CodeLanguage string

const (
	CodeLanguageUnspecified CodeLanguage = "LANGUAGE_UNSPECIFIED"
	CodeLanguagePython      CodeLanguage = "PYTHON"
)

// CodeExecutionResult holds the outcome and output of running model-generated code.
type CodeExecutionResult struct {
	Outcome CodeOutcome `json:"outcome,omitempty"`
	Output  string      `json:"output,omitempty"`
}

// CodeOutcome indicates how code execution ended.
type CodeOutcome string

const (
	CodeOutcomeUnspecified      CodeOutcome = "OUTCOME_UNSPECIFIED"
	CodeOutcomeOK               CodeOutcome = "OUTCOME_OK"
	CodeOutcomeFailed           CodeOutcome = "OUTCOME_FAILED"
	CodeOutcomeDeadlineExceeded CodeOutcome = "OUTCOME_DEADLINE_EXCEEDED"
)

// VideoMetadata describes a video segment's timing and frame rate.
type VideoMetadata struct {
	StartOffset time.Duration `json:"startOffset,omitempty"`
	EndOffset   time.Duration `json:"endOffset,omitempty"`
	FPS         *float64      `json:"fps,omitempty"`
}

// ---------------------------------------------------------------------------
// Tool Schema Types
// ---------------------------------------------------------------------------

// SchemaType is a JSON Schema primitive type used to define tool parameter shapes.
type SchemaType string

const (
	TypeString  SchemaType = "string"
	TypeNumber  SchemaType = "number"
	TypeInteger SchemaType = "integer"
	TypeBoolean SchemaType = "boolean"
	TypeArray   SchemaType = "array"
	TypeObject  SchemaType = "object"
)

// Schema defines the shape of tool parameters or structured response JSON.
type Schema struct {
	Type        SchemaType
	Description string
	Enum        []string
	Properties  map[string]*Schema
	Required    []string
	Items       *Schema
	Format      string
	Nullable    bool
}

// FunctionDeclaration describes a callable tool that can be offered to the model.
type FunctionDeclaration struct {
	Name        string
	Description string
	Parameters  *Schema
}

// ---------------------------------------------------------------------------
// Content Constructors
// ---------------------------------------------------------------------------

// NewTextContent creates a Content with the given role and a single text Part.
func NewTextContent(role Role, text string) *Content {
	return &Content{
		Role:  role,
		Parts: []*Part{{Text: text}},
	}
}

// NewFunctionCallContent returns model Content containing one or more function calls.
func NewFunctionCallContent(calls ...*FunctionCall) *Content {
	parts := make([]*Part, len(calls))
	for i, c := range calls {
		parts[i] = &Part{FunctionCall: c}
	}
	return &Content{Role: RoleModel, Parts: parts}
}

// NewFunctionResponseContent returns tool Content containing one or more function results.
func NewFunctionResponseContent(responses ...*FunctionResponse) *Content {
	parts := make([]*Part, len(responses))
	for i, r := range responses {
		parts[i] = &Part{FunctionResponse: r}
	}
	return &Content{Role: RoleTool, Parts: parts}
}

// NewUserContent returns user Content built from the given parts.
func NewUserContent(parts ...*Part) *Content {
	return &Content{Role: RoleUser, Parts: parts}
}

// NewModelContent returns model Content built from the given parts.
func NewModelContent(parts ...*Part) *Content {
	return &Content{Role: RoleModel, Parts: parts}
}

// ---------------------------------------------------------------------------
// Part Constructors
// ---------------------------------------------------------------------------

// TextPart returns a Part containing only plain text.
func TextPart(text string) *Part {
	return &Part{Text: text}
}

// FunctionCallPart returns a Part representing a single function call.
func FunctionCallPart(id, name string, args map[string]any) *Part {
	return &Part{FunctionCall: &FunctionCall{ID: id, Name: name, Args: args}}
}

// FunctionResponsePart returns a Part representing a single function result.
func FunctionResponsePart(id, name string, resp map[string]any) *Part {
	return &Part{FunctionResponse: &FunctionResponse{ID: id, Name: name, Response: resp}}
}

// BlobPart returns a Part with inline binary data and the given MIME type.
func BlobPart(mimeType string, data []byte) *Part {
	return &Part{InlineData: &Blob{MIMEType: mimeType, Data: data}}
}

// FileDataPart returns a Part that references data by URI.
func FileDataPart(mimeType, fileURI string) *Part {
	return &Part{FileData: &FileData{MIMEType: mimeType, FileURI: fileURI}}
}

// ---------------------------------------------------------------------------
// Text Extraction Helpers
// ---------------------------------------------------------------------------

// ExtractText returns the text from the first candidate of a RunResult.
func ExtractText(r *RunResult) string {
	if r == nil || r.Response == nil || len(r.Response.Candidates) == 0 {
		return ""
	}
	return ExtractResponseText(r.Response)
}

// ExtractResponseText returns the text from the first candidate of a Response.
func ExtractResponseText(resp *Response) string {
	if resp == nil || len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return ""
	}
	var parts []string
	for _, p := range resp.Candidates[0].Content.Parts {
		if p.Text != "" {
			parts = append(parts, p.Text)
		}
	}
	return strings.Join(parts, "\n")
}
