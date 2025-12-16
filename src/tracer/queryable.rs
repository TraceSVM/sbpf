//! Queryable trace format for AI analysis.
//!
//! This module provides a self-documenting, indexed trace format designed
//! for efficient querying by AI systems. The format includes:
//!
//! - Schema description explaining how to navigate the trace
//! - Function index for quick lookup by name or PC
//! - High-level summaries before detailed data
//! - Separated sections that can be read independently

use super::types::*;
use super::cfg::ControlFlowGraph;
use super::dataflow::DataFlowState;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Schema description for AI systems to understand the trace format.
pub const SCHEMA_DESCRIPTION: &str = r#"
TRACE FILE SCHEMA
=================

This JSON file contains a semantic execution trace of a Solana BPF program.
It is structured for efficient querying - read the summary first, then drill
down into specific functions or events as needed.

TOP-LEVEL STRUCTURE:
{
  "_schema": "Description of this file format (this text)",
  "program_id": "Base58 public key of the executed program",
  "execution_summary": {
    "result": "success" | "error",
    "total_compute_units": <number>,
    "total_instructions_executed": <number>,
    "functions_called": ["list", "of", "function", "names"],
    "syscalls_made": ["list", "of", "syscall", "names"],
    "accounts_accessed": [<account_index>, ...],
    "error_message": "only present if result is error"
  },
  "function_index": {
    "<function_name>": {
      "pc": <entry_pc>,
      "call_count": <times_called>,
      "total_instructions": <count>,
      "calls_functions": ["other", "functions"],
      "syscalls": ["syscalls", "used"],
      "memory_regions_accessed": ["Stack", "Input", ...],
      "summary": "Human-readable description of what this function does"
    }
  },
  "functions": {
    "<function_name>": <FunctionTrace object with full details>
  },
  "control_flow_graph": { ... CFG if enabled ... },
  "dataflow_analysis": { ... dataflow if enabled ... }
}

QUERYING GUIDE:
- To understand what the program does: Read execution_summary
- To find a specific function: Look up in function_index, then functions[name]
- To trace data flow: Check dataflow_analysis.definitions and .uses
- To find loops: Check control_flow_graph.loops
- To see all syscalls: Check execution_summary.syscalls_made, then search functions

FUNCTION TRACE STRUCTURE:
{
  "name": "function name or <unknown_XXXX> for unnamed",
  "entry_pc": <program counter at entry>,
  "call_sites": [<list of PCs that call this function>],
  "instructions": [
    {
      "pc": <program counter>,
      "mnemonic": "disassembled instruction",
      "semantic": "high-level description (for key instructions)"
    }
  ],
  "memory_accesses": [
    {
      "action": "load" | "store",
      "region": "Stack" | "Heap" | "Input" | "Rodata",
      "address": <vm_address>,
      "size": <bytes>,
      "value": "hex string",
      "semantic": "e.g., 'read account[0] data at offset 0'"
    }
  ],
  "syscalls": [
    {
      "name": "sol_log",
      "args": { "message": "decoded args" },
      "return_value": <if applicable>,
      "compute_units": <cost>
    }
  ],
  "child_calls": ["functions", "called", "from", "here"]
}

MEMORY REGIONS:
- Stack: Local variables, function arguments, return addresses
- Heap: Dynamically allocated memory (rare in Solana programs)
- Input: Account data passed to the program (accounts[N].data)
- Rodata: Read-only data, string constants, program constants
- Bytecode: The program's own code (usually not accessed as data)

COMMON ANALYSIS TASKS:
1. "What accounts does this program read/write?"
   → Check execution_summary.accounts_accessed
   → Search memory_accesses where region="Input"

2. "What does function X do?"
   → Read function_index["X"].summary
   → For details: functions["X"]

3. "Are there any loops?"
   → Check control_flow_graph.loops

4. "Where does value X come from?"
   → Check dataflow_analysis.definitions[X].origin

5. "What syscalls are made?"
   → execution_summary.syscalls_made for list
   → functions[name].syscalls for details per function
"#;

/// Root structure for queryable trace output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryableTrace {
    /// Schema description for AI systems.
    #[serde(rename = "_schema")]
    pub schema: String,

    /// Program identifier.
    pub program_id: Option<String>,

    /// High-level execution summary - read this first.
    pub execution_summary: ExecutionSummary,

    /// Function index for quick lookup.
    pub function_index: BTreeMap<String, FunctionIndexEntry>,

    /// Full function traces keyed by name.
    pub functions: BTreeMap<String, FunctionTrace>,

    /// Control flow graph (if enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_flow_graph: Option<ControlFlowGraph>,

    /// Dataflow analysis (if enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataflow_analysis: Option<DataFlowState>,
}

/// High-level execution summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSummary {
    /// Whether execution succeeded.
    pub result: String,
    /// Total compute units consumed.
    pub total_compute_units: u64,
    /// Total instructions executed.
    pub total_instructions_executed: u64,
    /// List of functions that were called.
    pub functions_called: Vec<String>,
    /// List of syscalls that were made.
    pub syscalls_made: Vec<String>,
    /// Account indices that were accessed.
    pub accounts_accessed: Vec<u64>,
    /// Error message if execution failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

/// Index entry for quick function lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionIndexEntry {
    /// Entry PC of the function.
    pub pc: u64,
    /// Number of times this function was called.
    pub call_count: u32,
    /// Total instructions executed in this function.
    pub total_instructions: u64,
    /// Functions called by this function.
    pub calls_functions: Vec<String>,
    /// Syscalls used by this function.
    pub syscalls: Vec<String>,
    /// Memory regions accessed.
    pub memory_regions_accessed: Vec<String>,
    /// Human-readable summary.
    pub summary: String,
}

/// Detailed function trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionTrace {
    /// Function name.
    pub name: String,
    /// Entry PC.
    pub entry_pc: u64,
    /// PCs of call sites that invoke this function.
    pub call_sites: Vec<u64>,
    /// Instructions executed (simplified).
    pub instructions: Vec<SimplifiedInstruction>,
    /// Memory accesses with semantic meaning.
    pub memory_accesses: Vec<SemanticMemoryAccess>,
    /// Syscalls made.
    pub syscalls: Vec<SyscallInfo>,
    /// Child functions called.
    pub child_calls: Vec<String>,
}

/// Simplified instruction for readability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedInstruction {
    /// Program counter.
    pub pc: u64,
    /// Disassembled mnemonic.
    pub mnemonic: String,
    /// High-level semantic description (for important instructions).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic: Option<String>,
}

/// Memory access with semantic context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMemoryAccess {
    /// Load or store.
    pub action: String,
    /// Memory region type.
    pub region: String,
    /// VM address.
    pub address: u64,
    /// Size in bytes.
    pub size: u64,
    /// Hex value.
    pub value: String,
    /// Semantic description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic: Option<String>,
}

/// Syscall information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyscallInfo {
    /// Syscall name.
    pub name: String,
    /// Decoded arguments.
    pub args: HashMap<String, String>,
    /// Return value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_value: Option<u64>,
    /// Compute units.
    pub compute_units: u64,
}

impl QueryableTrace {
    /// Convert a TraceContext into a QueryableTrace.
    ///
    /// Set `include_instructions` to false to create a compact trace
    /// without per-instruction details (much smaller file size).
    pub fn from_trace_context(ctx: &TraceContext) -> Self {
        Self::from_trace_context_with_options(ctx, true)
    }

    /// Convert with options for controlling output size.
    pub fn from_trace_context_compact(ctx: &TraceContext) -> Self {
        Self::from_trace_context_with_options(ctx, false)
    }

    fn from_trace_context_with_options(ctx: &TraceContext, include_instructions: bool) -> Self {
        let mut functions: BTreeMap<String, FunctionTrace> = BTreeMap::new();
        let mut function_index: BTreeMap<String, FunctionIndexEntry> = BTreeMap::new();
        let mut all_syscalls: Vec<String> = Vec::new();
        let mut all_functions: Vec<String> = Vec::new();
        let mut total_instructions: u64 = 0;
        let mut accounts_accessed: Vec<u64> = Vec::new();

        // Process execution tree recursively
        for frame in &ctx.execution_tree {
            Self::process_frame(
                frame,
                &mut functions,
                &mut function_index,
                &mut all_syscalls,
                &mut all_functions,
                &mut total_instructions,
                &mut accounts_accessed,
                None,
                include_instructions,
            );
        }

        // Deduplicate
        all_syscalls.sort();
        all_syscalls.dedup();
        all_functions.sort();
        all_functions.dedup();
        accounts_accessed.sort();
        accounts_accessed.dedup();

        // Build execution summary
        let (result, error_message) = match &ctx.result {
            TraceResult::Success { .. } => ("success".to_string(), None),
            TraceResult::Error { message } => ("error".to_string(), Some(message.clone())),
        };

        let execution_summary = ExecutionSummary {
            result,
            total_compute_units: ctx.total_compute_units,
            total_instructions_executed: total_instructions,
            functions_called: all_functions,
            syscalls_made: all_syscalls,
            accounts_accessed,
            error_message,
        };

        // Generate function summaries
        for (name, entry) in function_index.iter_mut() {
            entry.summary = Self::generate_function_summary(name, entry, functions.get(name));
        }

        QueryableTrace {
            schema: SCHEMA_DESCRIPTION.to_string(),
            program_id: ctx.program_id.clone(),
            execution_summary,
            function_index,
            functions,
            control_flow_graph: ctx.control_flow_graph.clone(),
            dataflow_analysis: ctx.dataflow.clone(),
        }
    }

    fn process_frame(
        frame: &TracedCallFrame,
        functions: &mut BTreeMap<String, FunctionTrace>,
        function_index: &mut BTreeMap<String, FunctionIndexEntry>,
        all_syscalls: &mut Vec<String>,
        all_functions: &mut Vec<String>,
        total_instructions: &mut u64,
        accounts_accessed: &mut Vec<u64>,
        parent_call_site: Option<u64>,
        include_instructions: bool,
    ) {
        let func_name = frame
            .symbol_name
            .clone()
            .unwrap_or_else(|| format!("<func_{:04x}>", frame.start_pc));

        all_functions.push(func_name.clone());

        // Collect data from this frame
        let mut instructions = Vec::new();
        let mut memory_accesses = Vec::new();
        let mut syscalls = Vec::new();
        let mut child_calls = Vec::new();
        let mut regions_accessed: Vec<String> = Vec::new();
        let mut syscall_names: Vec<String> = Vec::new();

        for event in &frame.events {
            match event {
                TraceEvent::Instruction(insn) => {
                    *total_instructions += 1;

                    // Check if this instruction is a syscall and extract name
                    if insn.mnemonic.starts_with("syscall ") {
                        let syscall_name = insn.mnemonic
                            .strip_prefix("syscall ")
                            .unwrap_or(&insn.mnemonic)
                            .to_string();
                        all_syscalls.push(syscall_name.clone());
                        syscall_names.push(syscall_name.clone());
                        // Record as a syscall info as well
                        syscalls.push(SyscallInfo {
                            name: syscall_name,
                            args: HashMap::new(),
                            return_value: None,
                            compute_units: insn.compute_units,
                        });
                    }

                    if include_instructions {
                        let semantic = Self::get_instruction_semantic(&insn.mnemonic);
                        instructions.push(SimplifiedInstruction {
                            pc: insn.pc,
                            mnemonic: insn.mnemonic.clone(),
                            semantic,
                        });
                    }
                }
                TraceEvent::MemoryAccess(mem) => {
                    let region_str = format!("{:?}", mem.region);
                    if !regions_accessed.contains(&region_str) {
                        regions_accessed.push(region_str.clone());
                    }

                    // Try to extract account index from Input region addresses
                    if mem.region == MemoryRegionType::Input {
                        let account_idx = Self::address_to_account_index(mem.address);
                        if let Some(idx) = account_idx {
                            accounts_accessed.push(idx);
                        }
                    }

                    if include_instructions {
                        let semantic = Self::get_memory_semantic(mem);
                        memory_accesses.push(SemanticMemoryAccess {
                            action: format!("{:?}", mem.action),
                            region: region_str,
                            address: mem.address,
                            size: mem.size,
                            value: mem.value_hex.clone(),
                            semantic,
                        });
                    }
                }
                TraceEvent::Syscall(sys) => {
                    all_syscalls.push(sys.name.clone());
                    syscall_names.push(sys.name.clone());
                    syscalls.push(SyscallInfo {
                        name: sys.name.clone(),
                        args: sys.args_decoded.clone(),
                        return_value: sys.return_value,
                        compute_units: sys.compute_units,
                    });
                }
                TraceEvent::FunctionCall(call) => {
                    let target = call
                        .target_function
                        .clone()
                        .unwrap_or_else(|| format!("<func_{:04x}>", call.target_pc));
                    child_calls.push(target);
                }
                TraceEvent::FunctionReturn(_) => {}
            }
        }

        // Process child frames
        for sub_frame in &frame.sub_calls {
            let sub_name = sub_frame
                .symbol_name
                .clone()
                .unwrap_or_else(|| format!("<func_{:04x}>", sub_frame.start_pc));
            child_calls.push(sub_name);

            Self::process_frame(
                sub_frame,
                functions,
                function_index,
                all_syscalls,
                all_functions,
                total_instructions,
                accounts_accessed,
                Some(frame.start_pc),
                include_instructions,
            );
        }

        child_calls.sort();
        child_calls.dedup();
        syscall_names.sort();
        syscall_names.dedup();

        // Build function trace
        let func_trace = FunctionTrace {
            name: func_name.clone(),
            entry_pc: frame.start_pc,
            call_sites: parent_call_site.into_iter().collect(),
            instructions,
            memory_accesses,
            syscalls,
            child_calls: child_calls.clone(),
        };

        // Update or create function entry
        if let Some(existing) = functions.get_mut(&func_name) {
            // Merge call sites
            if let Some(site) = parent_call_site {
                if !existing.call_sites.contains(&site) {
                    existing.call_sites.push(site);
                }
            }
        } else {
            functions.insert(func_name.clone(), func_trace);
        }

        // Build index entry
        let index_entry = FunctionIndexEntry {
            pc: frame.start_pc,
            call_count: 1, // Will be incremented on subsequent calls
            total_instructions: *total_instructions,
            calls_functions: child_calls,
            syscalls: syscall_names,
            memory_regions_accessed: regions_accessed,
            summary: String::new(), // Filled in later
        };

        if let Some(existing) = function_index.get_mut(&func_name) {
            existing.call_count += 1;
        } else {
            function_index.insert(func_name, index_entry);
        }
    }

    /// Attempt to map a memory address to an account index.
    /// This is Solana-specific: input region layout.
    fn address_to_account_index(address: u64) -> Option<u64> {
        // Input region starts at 0x400000000 (17179869184)
        const INPUT_START: u64 = 0x400000000;
        if address >= INPUT_START {
            // Very rough heuristic - actual mapping depends on account layout
            // First account typically at INPUT_START + small offset
            // This would need to be refined with actual account metadata
            Some(0) // For now, assume account 0
        } else {
            None
        }
    }

    /// Generate semantic description for key instructions.
    fn get_instruction_semantic(mnemonic: &str) -> Option<String> {
        // Add semantic meaning to important instruction patterns
        if mnemonic.starts_with("call ") {
            Some("function call".to_string())
        } else if mnemonic.starts_with("exit") {
            Some("return from program".to_string())
        } else if mnemonic.contains("jeq") || mnemonic.contains("jne") || mnemonic.contains("jgt") {
            Some("conditional branch".to_string())
        } else if mnemonic.starts_with("ja ") {
            Some("unconditional jump".to_string())
        } else if mnemonic.starts_with("ldxdw") || mnemonic.starts_with("ldxw") {
            Some("load from memory".to_string())
        } else if mnemonic.starts_with("stxdw") || mnemonic.starts_with("stxw") {
            Some("store to memory".to_string())
        } else {
            None
        }
    }

    /// Generate semantic description for memory accesses.
    fn get_memory_semantic(mem: &MemoryAccessEvent) -> Option<String> {
        match mem.region {
            MemoryRegionType::Input => {
                let action = if mem.action == MemoryAccessType::Load {
                    "read"
                } else {
                    "write"
                };
                Some(format!(
                    "{} {} bytes from account data at offset 0x{:x}",
                    action, mem.size, mem.address & 0xFFFF
                ))
            }
            MemoryRegionType::Stack => {
                if mem.size == 8 {
                    Some("stack variable (likely pointer or u64)".to_string())
                } else if mem.size == 4 {
                    Some("stack variable (likely u32)".to_string())
                } else {
                    None
                }
            }
            MemoryRegionType::Rodata => Some("read constant data".to_string()),
            _ => None,
        }
    }

    /// Generate a human-readable summary for a function.
    fn generate_function_summary(
        name: &str,
        index: &FunctionIndexEntry,
        trace: Option<&FunctionTrace>,
    ) -> String {
        let mut parts = Vec::new();

        // Basic info
        if name == "entrypoint" {
            parts.push("Program entry point".to_string());
        }

        // Syscall summary
        if !index.syscalls.is_empty() {
            if index.syscalls.contains(&"sol_log".to_string())
                || index.syscalls.contains(&"sol_log_".to_string())
            {
                parts.push("logs messages".to_string());
            }
            if index.syscalls.contains(&"sol_invoke_signed".to_string()) {
                parts.push("makes CPI calls".to_string());
            }
        }

        // Memory summary
        if index
            .memory_regions_accessed
            .contains(&"Input".to_string())
        {
            parts.push("accesses account data".to_string());
        }

        // Call summary
        if !index.calls_functions.is_empty() {
            parts.push(format!("calls {} functions", index.calls_functions.len()));
        }

        if parts.is_empty() {
            "Internal function".to_string()
        } else {
            parts.join(", ")
        }
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Serialize to compact JSON.
    pub fn to_json_compact(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_included() {
        let ctx = TraceContext {
            program_id: Some("test".to_string()),
            execution_tree: vec![],
            total_compute_units: 0,
            result: TraceResult::Success { return_value: 0 },
            control_flow_graph: None,
            dataflow: None,
        };

        let queryable = QueryableTrace::from_trace_context(&ctx);
        assert!(queryable.schema.contains("TRACE FILE SCHEMA"));
        assert!(queryable.schema.contains("QUERYING GUIDE"));
    }
}
