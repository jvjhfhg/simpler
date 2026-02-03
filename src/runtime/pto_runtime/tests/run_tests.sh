#!/bin/bash
#
# PTO Runtime Test Runner
# Automatically compiles, runs, reports, and cleans up test executables.
#
# Usage:
#   ./run_tests.sh          # Run all phase tests
#   ./run_tests.sh phase1   # Run only phase 1 tests
#   ./run_tests.sh phase4   # Run only phase 4 tests
#   ./run_tests.sh -v       # Verbose mode (show full output)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$SCRIPT_DIR/../runtime"
PLATFORM_INCLUDE_DIR="$SCRIPT_DIR/../../../../src/platform/include"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
COMPILER="g++"
CFLAGS="-std=c++17 -I$RUNTIME_DIR -I$PLATFORM_INCLUDE_DIR -Wall -Wextra"
RUNTIME_SRC="$RUNTIME_DIR/runtime.cpp $RUNTIME_DIR/tensor_descriptor.cpp"

# Test files (in execution order)
declare -a TEST_FILES=(
    "test_phase1_state_machine.cpp"
    "test_phase2_scope_end.cpp"
    "test_phase3_ring_buffers.cpp"
    "test_phase4_shared_header.cpp"
)

# Parse arguments
VERBOSE=false
FILTER=""

for arg in "$@"; do
    case $arg in
        -v|--verbose)
            VERBOSE=true
            ;;
        phase*)
            FILTER="$arg"
            ;;
        -h|--help)
            echo "PTO Runtime Test Runner"
            echo ""
            echo "Usage: $0 [options] [filter]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Show full test output"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Filters:"
            echo "  phase1           Run only Phase 1 tests"
            echo "  phase2           Run only Phase 2 tests"
            echo "  phase3           Run only Phase 3 tests"
            echo "  phase4           Run only Phase 4 tests"
            echo ""
            echo "Examples:"
            echo "  $0               Run all tests"
            echo "  $0 -v            Run all tests with verbose output"
            echo "  $0 phase4        Run only Phase 4 tests"
            echo "  $0 -v phase4     Run Phase 4 tests with verbose output"
            exit 0
            ;;
    esac
done

# Track results
declare -a EXECUTABLES=()
TOTAL_PASSED=0
TOTAL_FAILED=0
declare -a PHASE_RESULTS=()

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}PTO Runtime Test Suite${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

cd "$SCRIPT_DIR"

# Function to compile a test
compile_test() {
    local src_file="$1"
    local exe_name="${src_file%.cpp}"

    echo -ne "  Compiling ${src_file}... "

    if $COMPILER $CFLAGS -o "$exe_name" "$src_file" $RUNTIME_SRC 2>/tmp/compile_error.txt; then
        echo -e "${GREEN}OK${NC}"
        EXECUTABLES+=("$exe_name")
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        if $VERBOSE; then
            cat /tmp/compile_error.txt
        fi
        return 1
    fi
}

# Function to run a test
run_test() {
    local exe_name="$1"
    local test_name="$(basename "$exe_name")"

    echo -e "\n${YELLOW}Running: $test_name${NC}"
    echo "------------------------------------------------------------"

    # Run test and capture output
    local output
    local exit_code

    if output=$("./$exe_name" 2>&1); then
        exit_code=0
    else
        exit_code=$?
    fi

    # Extract results from output
    local results_line=$(echo "$output" | grep -E "Results: [0-9]+ passed, [0-9]+ failed" | tail -1)

    if [[ -n "$results_line" ]]; then
        local passed=$(echo "$results_line" | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+")
        local failed=$(echo "$results_line" | grep -oE "[0-9]+ failed" | grep -oE "[0-9]+")

        TOTAL_PASSED=$((TOTAL_PASSED + passed))
        TOTAL_FAILED=$((TOTAL_FAILED + failed))

        if [[ "$failed" -eq 0 ]]; then
            PHASE_RESULTS+=("${GREEN}✓${NC} $test_name: ${GREEN}$passed passed${NC}")
        else
            PHASE_RESULTS+=("${RED}✗${NC} $test_name: ${GREEN}$passed passed${NC}, ${RED}$failed failed${NC}")
        fi

        if $VERBOSE; then
            echo "$output"
        else
            # Show condensed output: just the test names and final results
            echo "$output" | grep -E "^(=== Test|  PASS:|  FAIL:|Results:)" | head -30
            local total_tests=$((passed + failed))
            if [[ $total_tests -gt 30 ]]; then
                echo "  ... ($total_tests total tests)"
            fi
        fi
    else
        PHASE_RESULTS+=("${RED}✗${NC} $test_name: ${RED}No results found${NC}")
        if $VERBOSE; then
            echo "$output"
        fi
    fi

    return $exit_code
}

# Compile phase
echo -e "${BLUE}Compiling tests...${NC}"

for test_file in "${TEST_FILES[@]}"; do
    # Apply filter if specified
    if [[ -n "$FILTER" ]]; then
        case $FILTER in
            phase1) [[ "$test_file" != *"phase1"* ]] && continue ;;
            phase2) [[ "$test_file" != *"phase2"* ]] && continue ;;
            phase3) [[ "$test_file" != *"phase3"* ]] && continue ;;
            phase4) [[ "$test_file" != *"phase4"* ]] && continue ;;
        esac
    fi

    if [[ -f "$test_file" ]]; then
        compile_test "$test_file" || true
    else
        echo -e "  ${YELLOW}Skipping $test_file (not found)${NC}"
    fi
done

# Run phase
if [[ ${#EXECUTABLES[@]} -eq 0 ]]; then
    echo -e "\n${RED}No tests to run!${NC}"
    exit 1
fi

echo -e "\n${BLUE}Running tests...${NC}"

for exe in "${EXECUTABLES[@]}"; do
    run_test "$exe" || true
done

# Summary
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}============================================================${NC}"

for result in "${PHASE_RESULTS[@]}"; do
    echo -e "  $result"
done

echo ""
echo -e "  ${BLUE}Total:${NC} ${GREEN}$TOTAL_PASSED passed${NC}, ${RED}$TOTAL_FAILED failed${NC}"

# Cleanup
echo -e "\n${BLUE}Cleaning up executables...${NC}"

for exe in "${EXECUTABLES[@]}"; do
    if [[ -f "$exe" ]]; then
        rm -f "$exe"
        echo "  Removed: $exe"
    fi
done

# Also clean up any stray test executables
for pattern in test_phase*; do
    if [[ -x "$pattern" && ! "$pattern" == *.cpp && ! "$pattern" == *.sh ]]; then
        rm -f "$pattern"
        echo "  Removed: $pattern"
    fi
done

echo ""
echo -e "${BLUE}============================================================${NC}"

# Exit with appropriate code
if [[ $TOTAL_FAILED -gt 0 ]]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
