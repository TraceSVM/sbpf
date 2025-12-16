//! Data structures for semantic execution tracing.
//!
//! These types represent the hierarchical trace output consumed by LLMs
//! for program analysis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Maximum bytes to capture for memory values before truncation.
pub const MAX_MEMORY_VALUE_BYTES: usize = 64;

/// Root trace context containing the full execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// Optional program identifier (e.g., base58 pubkey).
    pub program_id: Option<String>,
    /// Hierarchical execution tree organized by call frames.
    pub execution_tree: Vec<TracedCallFrame>,
    /// Total compute units consumed during execution.
    pub total_compute_units: u64,
    /// Final execution result.
    pub result: TraceResult,
    /// Text section virtual address (for converting PC to ELF addresses).
    pub text_section_vaddr: u64,
    /// Control flow graph analysis (if enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_flow_graph: Option<super::cfg::ControlFlowGraph>,
    /// Data flow analysis state (if enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataflow: Option<super::dataflow::DataFlowState>,
}

/// Result of traced execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum TraceResult {
    /// Successful execution with return value.
    Success {
        /// Return value in r0.
        return_value: u64,
    },
    /// Execution error.
    Error {
        /// Error description.
        message: String,
    },
}

/// A traced function call frame (hierarchical).
///
/// Each frame represents a function scope, containing the events that
/// occurred within that function and any nested function calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracedCallFrame {
    /// Resolved symbol name (from symbol_map) or None.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol_name: Option<String>,
    /// PC at frame entry.
    pub start_pc: u64,
    /// Call depth (0 = entrypoint).
    pub depth: u32,
    /// Events recorded within this frame.
    pub events: Vec<TraceEvent>,
    /// Nested function calls made from this frame.
    pub sub_calls: Vec<TracedCallFrame>,
    /// Compute units consumed in this frame (excluding sub-calls).
    pub frame_compute_units: u64,
}

impl TracedCallFrame {
    /// Create a new call frame.
    pub fn new(symbol_name: Option<String>, start_pc: u64, depth: u32) -> Self {
        Self {
            symbol_name,
            start_pc,
            depth,
            events: Vec::new(),
            sub_calls: Vec::new(),
            frame_compute_units: 0,
        }
    }
}

/// Individual trace event types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TraceEvent {
    /// Instruction execution event.
    Instruction(InstructionEvent),
    /// Memory load/store event.
    MemoryAccess(MemoryAccessEvent),
    /// Syscall invocation event.
    Syscall(SyscallEvent),
    /// Internal function call event.
    FunctionCall(FunctionCallEvent),
    /// Function return event.
    FunctionReturn(FunctionReturnEvent),
}

/// Instruction execution details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionEvent {
    /// Program counter.
    pub pc: u64,
    /// Raw opcode byte.
    pub opcode: u8,
    /// Disassembled mnemonic (e.g., "add64 r1, r2").
    pub mnemonic: String,
    /// Register changes: register name -> new value.
    /// Only registers that changed are included.
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub regs_diff: HashMap<String, u64>,
    /// Compute units consumed by this instruction.
    pub compute_units: u64,
}

/// Memory region classification for semantic understanding.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MemoryRegionType {
    /// Read-only data section (constants, strings).
    Rodata,
    /// Program bytecode.
    Bytecode,
    /// Stack memory (local variables, return addresses).
    Stack,
    /// Heap memory (dynamic allocations).
    Heap,
    /// Input/parameter data (account data in Solana).
    Input,
    /// Unknown or invalid memory region.
    Unknown,
}

impl std::fmt::Display for MemoryRegionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryRegionType::Rodata => write!(f, "rodata"),
            MemoryRegionType::Bytecode => write!(f, "bytecode"),
            MemoryRegionType::Stack => write!(f, "stack"),
            MemoryRegionType::Heap => write!(f, "heap"),
            MemoryRegionType::Input => write!(f, "input"),
            MemoryRegionType::Unknown => write!(f, "unknown"),
        }
    }
}

/// Memory access type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemoryAccessType {
    /// Read from memory.
    Load,
    /// Write to memory.
    Store,
}

/// Memory access event details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAccessEvent {
    /// Classified memory region.
    pub region: MemoryRegionType,
    /// Load or Store.
    pub action: MemoryAccessType,
    /// VM address accessed.
    pub address: u64,
    /// Size in bytes.
    pub size: u64,
    /// Hex-encoded value (truncated if > MAX_MEMORY_VALUE_BYTES).
    pub value_hex: String,
    /// Whether value was truncated.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

/// Syscall invocation event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyscallEvent {
    /// Syscall name (e.g., "sol_log", "sol_memcpy").
    pub name: String,
    /// Decoded arguments with semantic names.
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub args_decoded: HashMap<String, String>,
    /// Raw register values r1-r5.
    pub raw_args: [u64; 5],
    /// Return value in r0 after syscall.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_value: Option<u64>,
    /// Compute units consumed.
    pub compute_units: u64,
}

/// Internal function call event.
///
/// This event is emitted when the program calls an internal function
/// (not a syscall). This is useful for decompilation as it shows
/// the control flow between functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallEvent {
    /// Target function name (if known from symbol map).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_function: Option<String>,
    /// Target PC (instruction offset) being called.
    pub target_pc: u64,
    /// PC of the call instruction.
    pub call_site_pc: u64,
    /// Whether this is a register-based call (callx) vs immediate call.
    pub is_indirect: bool,
    /// Arguments passed in r1-r5 at time of call.
    pub args: [u64; 5],
    /// New call depth after this call.
    pub depth: u32,
}

/// Function return event.
///
/// This event is emitted when returning from an internal function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionReturnEvent {
    /// Function name being returned from (if known).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_function: Option<String>,
    /// Return value in r0.
    pub return_value: u64,
    /// PC being returned to.
    pub return_pc: u64,
    /// Call depth after return.
    pub depth: u32,
}
