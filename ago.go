// Package ago is a Go framework for building AI agents.
// All core types live here — no separate types package.
package ago

import (
	"context"
	"iter"
	"time"
)

// ---------------------------------------------------------------------------
// Roles
// ---------------------------------------------------------------------------

// Role identifies the sender of a message in a conversation (user, model, system, or tool).
type Role string

const (
	RoleUser   Role = "user"   // Message from the end user.
	RoleModel  Role = "model"  // Message from the LLM.
	RoleSystem Role = "system" // System instruction (often passed via GenerateConfig).
	RoleTool   Role = "tool"   // Message carrying tool/function call results.
)

// ---------------------------------------------------------------------------
// Content & Parts
// ---------------------------------------------------------------------------

// Content is a single turn in a conversation: a role plus one or more parts (text, tool calls, media, etc.).
type Content struct {
	Role  Role    `json:"role,omitempty"`
	Parts []*Part `json:"parts,omitempty"`
}

// Part is one segment of a message. At most one of Text, FunctionCall, FunctionResponse,
// InlineData, FileData, ExecutableCode, or CodeExecutionResult should be set.
// Thought and ThoughtSignature apply when the part represents model reasoning.
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

// FunctionCall represents the model requesting a tool/function invocation (name and arguments).
type FunctionCall struct {
	ID   string         `json:"id,omitempty"`
	Name string         `json:"name,omitempty"`
	Args map[string]any `json:"args,omitempty"`
}

// FunctionResponse is the result of a tool invocation, sent back to the model for the next turn.
type FunctionResponse struct {
	ID       string         `json:"id,omitempty"`
	Name     string         `json:"name,omitempty"`
	Response map[string]any `json:"response,omitempty"`
}

// Blob holds inline binary data (e.g. images, audio) with a MIME type.
type Blob struct {
	MIMEType string `json:"mimeType,omitempty"`
	Data     []byte `json:"data,omitempty"`
}

// FileData references external data by URI (e.g. GCS or uploaded file), with optional display name.
type FileData struct {
	MIMEType    string `json:"mimeType,omitempty"`
	FileURI     string `json:"fileUri,omitempty"`
	DisplayName string `json:"displayName,omitempty"`
}

// ExecutableCode is code produced by the model for execution (e.g. in code execution flows).
type ExecutableCode struct {
	Code     string       `json:"code,omitempty"`
	Language CodeLanguage `json:"language,omitempty"`
}

// CodeLanguage is the language of executable code (e.g. PYTHON).
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

// CodeOutcome indicates how code execution ended (ok, failed, or deadline exceeded).
type CodeOutcome string

const (
	CodeOutcomeUnspecified      CodeOutcome = "OUTCOME_UNSPECIFIED"
	CodeOutcomeOK               CodeOutcome = "OUTCOME_OK"
	CodeOutcomeFailed           CodeOutcome = "OUTCOME_FAILED"
	CodeOutcomeDeadlineExceeded CodeOutcome = "OUTCOME_DEADLINE_EXCEEDED"
)

// VideoMetadata describes a video segment (start/end offsets and optional FPS) for inline or file video.
type VideoMetadata struct {
	StartOffset time.Duration `json:"startOffset,omitempty"`
	EndOffset   time.Duration `json:"endOffset,omitempty"`
	FPS         *float64      `json:"fps,omitempty"`
}

// ---------------------------------------------------------------------------
// Token Usage (aligned with GenerateContentResponseUsageMetadata)
// ---------------------------------------------------------------------------

// ModalityTokenCount is token count per modality (TEXT, IMAGE, etc.).
type ModalityTokenCount struct {
	Modality   string `json:"modality,omitempty"`
	TokenCount int32  `json:"tokenCount,omitempty"`
}

// TokenUsage reports full token usage: prompt, completion, cache, thought, tool use, and per-modality details.
type TokenUsage struct {
	PromptTokenCount           int32                 `json:"promptTokenCount,omitempty"`
	CandidatesTokenCount       int32                 `json:"candidatesTokenCount,omitempty"`
	TotalTokenCount            int32                 `json:"totalTokenCount,omitempty"`
	CachedContentTokenCount    int32                 `json:"cachedContentTokenCount,omitempty"`
	ThoughtsTokenCount         int32                 `json:"thoughtsTokenCount,omitempty"`
	ToolUsePromptTokenCount    int32                 `json:"toolUsePromptTokenCount,omitempty"`
	TrafficType                string                `json:"trafficType,omitempty"`
	CacheTokensDetails         []*ModalityTokenCount `json:"cacheTokensDetails,omitempty"`
	CandidatesTokensDetails    []*ModalityTokenCount `json:"candidatesTokensDetails,omitempty"`
	PromptTokensDetails        []*ModalityTokenCount `json:"promptTokensDetails,omitempty"`
	ToolUsePromptTokensDetails []*ModalityTokenCount `json:"toolUsePromptTokensDetails,omitempty"`
}

// ---------------------------------------------------------------------------
// Finish Reason
// ---------------------------------------------------------------------------

// FinishReason explains why the model stopped (natural stop, limit, tool call, safety, or error).
type FinishReason string

const (
	FinishReasonStop      FinishReason = "stop"       // Model finished normally.
	FinishReasonMaxTokens FinishReason = "max_tokens" // Hit output token limit.
	FinishReasonToolCall  FinishReason = "tool_call"  // Stopped to request a tool call.
	FinishReasonSafety    FinishReason = "safety"     // Blocked by safety filters.
	FinishReasonError     FinishReason = "error"      // Error during generation.
)

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

// Response is the result of a non-streaming Generate call: one or more candidates plus usage and model version.
type Response struct {
	Candidates   []*Candidate
	Usage        TokenUsage
	ModelVersion string
}

// Candidate is one possible generation (content, finish reason, index; aligned with genai Candidate).
type Candidate struct {
	Content       *Content     `json:"content,omitempty"`
	FinishReason  FinishReason `json:"finishReason,omitempty"`
	Index         int          `json:"index,omitempty"`
	TokenCount    int32        `json:"tokenCount,omitempty"`
	FinishMessage string       `json:"finishMessage,omitempty"`
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

// StreamChunk is a single item from GenerateStream. Sent as SSE: candidates, usageMetadata, complete.
// No event/modelVersion/responseId/flat content. Last chunk has Complete: true with full candidates and usageMetadata (source of truth).
type StreamChunk struct {
	Candidates   []*Candidate `json:"candidates,omitempty"`
	Usage        *TokenUsage  `json:"usageMetadata,omitempty"`
	Complete     bool         `json:"complete,omitempty"` // true on final event only
	ErrorMessage string       `json:"error,omitempty"`
}

// ---------------------------------------------------------------------------
// Tool Schema Types
// ---------------------------------------------------------------------------

// SchemaType is a JSON Schema primitive type for tool parameters and structured outputs.
type SchemaType string

const (
	TypeString  SchemaType = "string"
	TypeNumber  SchemaType = "number"
	TypeInteger SchemaType = "integer"
	TypeBoolean SchemaType = "boolean"
	TypeArray   SchemaType = "array"
	TypeObject  SchemaType = "object"
)

// Schema defines the shape of tool parameters or response JSON (type, properties, required, enum, etc.).
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

// FunctionDeclaration declares a callable tool: name, description, and parameter schema.
type FunctionDeclaration struct {
	Name        string
	Description string
	Parameters  *Schema
}

// ---------------------------------------------------------------------------
// GenerateConfig
// ---------------------------------------------------------------------------

// GenerateConfig holds all optional parameters for a generate request (sampling, tools, system instruction, etc.).
type GenerateConfig struct {
	MaxOutputTokens   int
	Temperature       *float64
	TopP              *float64
	TopK              *float64
	StopSequences     []string
	SystemInstruction *Content
	Tools             []*FunctionDeclaration
	ResponseMIMEType  string
	ResponseSchema    *Schema
	ThinkingConfig    *ThinkingConfig
	Seed              *int
	PresencePenalty   *float64
	FrequencyPenalty  *float64
}

// ThinkingConfig configures extended/reasoning tokens (e.g. thinking budget) when the model supports it.
type ThinkingConfig struct {
	Enabled bool
	Budget  int
}

// ---------------------------------------------------------------------------
// LLM Interface
// ---------------------------------------------------------------------------

// LLM is the interface implemented by all LLM backends (e.g. GenAI, OpenAI).
// Callers use Name to identify the backend, Generate/GenerateStream for completion, and Close for cleanup.
type LLM interface {
	Name() string
	Generate(ctx context.Context, model string, contents []*Content, config *GenerateConfig) (*Response, error)
	GenerateStream(ctx context.Context, model string, contents []*Content, config *GenerateConfig) iter.Seq2[*StreamChunk, error]
	Close() error
}

// ---------------------------------------------------------------------------
// Content Constructors
// ---------------------------------------------------------------------------

// NewTextContent returns a Content with the given role and a single text part.
func NewTextContent(role Role, text string) *Content {
	return &Content{
		Role:  role,
		Parts: []*Part{{Text: text}},
	}
}

// NewFunctionCallContent returns model Content containing one or more function calls (for tool-call turns).
func NewFunctionCallContent(calls ...*FunctionCall) *Content {
	parts := make([]*Part, len(calls))
	for i, c := range calls {
		parts[i] = &Part{FunctionCall: c}
	}
	return &Content{Role: RoleModel, Parts: parts}
}

// NewFunctionResponseContent returns tool Content containing one or more function results (for tool-response turns).
func NewFunctionResponseContent(responses ...*FunctionResponse) *Content {
	parts := make([]*Part, len(responses))
	for i, r := range responses {
		parts[i] = &Part{FunctionResponse: r}
	}
	return &Content{Role: RoleTool, Parts: parts}
}

// NewUserContent returns user Content built from the given parts (text, blobs, file refs, etc.).
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

// FunctionCallPart returns a Part representing a single function call (id, name, args).
func FunctionCallPart(id, name string, args map[string]any) *Part {
	return &Part{FunctionCall: &FunctionCall{ID: id, Name: name, Args: args}}
}

// FunctionResponsePart returns a Part representing a single function result (id, name, response).
func FunctionResponsePart(id, name string, resp map[string]any) *Part {
	return &Part{FunctionResponse: &FunctionResponse{ID: id, Name: name, Response: resp}}
}

// BlobPart returns a Part with inline binary data (e.g. image) and the given MIME type.
func BlobPart(mimeType string, data []byte) *Part {
	return &Part{InlineData: &Blob{MIMEType: mimeType, Data: data}}
}

// FileDataPart returns a Part that references data by URI (e.g. GCS or uploaded file).
func FileDataPart(mimeType, fileURI string) *Part {
	return &Part{FileData: &FileData{MIMEType: mimeType, FileURI: fileURI}}
}
