"""
PTO Runtime Build Configuration

This configuration file is discovered by RuntimeBuilder to compile the PTO runtime
for three execution contexts: AICore (compute), AICPU (scheduler), and Host (orchestrator).
"""

BUILD_CONFIG = {
    "aicore": {
        "include_dirs": ["runtime"],
        "source_dirs": ["aicore", "runtime"]
    },
    "aicpu": {
        "include_dirs": ["runtime"],
        "source_dirs": ["aicpu", "runtime"]
    },
    "host": {
        "include_dirs": ["runtime"],
        "source_dirs": ["host", "runtime"]
    }
}
