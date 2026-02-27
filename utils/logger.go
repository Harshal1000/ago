package utils

import (
	"os"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// Logger is the package-level logger instance. It is initialized automatically via init()
// with sensible defaults. Call InitLogger() to re-initialize with custom settings, or
// use zap.L() anywhere (since ReplaceGlobals is called).
var Logger *zap.Logger

// initializes the global logger.
//
// By default (ENV != "dev"), it uses a colored development config for human-readable output.
// When ENV=dev, it uses structured JSON production config.
//
// Safe to call multiple times — each call replaces the global logger.
func init() {
	var config zap.Config

	if os.Getenv("ENV") == "dev" {
		config = zap.NewProductionConfig()
		config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	} else {
		config = zap.NewDevelopmentConfig()
		config.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
	}

	config.DisableStacktrace = true

	config.EncoderConfig.EncodeTime = func(t time.Time, enc zapcore.PrimitiveArrayEncoder) {
		enc.AppendString(t.Format("15:04:05.000"))
	}

	l, err := config.Build()
	if err != nil {
		panic(err)
	}

	Logger = l
	zap.ReplaceGlobals(Logger)
}

// SyncLogger flushes any buffered log entries. Call this on application exit via defer.
func SyncLogger() {
	if Logger != nil {
		_ = Logger.Sync()
	}
}
