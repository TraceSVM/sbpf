// Example: Semantic Execution Tracing for Solana Programs
//
// This example demonstrates how to use the semantic tracer to generate
// detailed execution traces suitable for LLM analysis and decompilation.
//
// Features demonstrated:
// - Basic execution tracing with instruction events and memory access
// - Control Flow Graph (CFG) analysis with basic blocks, edges, and loops
// - Data Flow analysis with value lineage, taint tracking, and def-use chains
//
// Run with: cargo run --example trace_program --features semantic-tracer

#![cfg(feature = "semantic-tracer")]

use solana_sbpf::{
    assembler::assemble,
    ebpf,
    elf::Executable,
    program::BuiltinProgram,
    tracer::Tracer,
    verifier::RequisiteVerifier,
    vm::{Config, EbpfVm},
};
use std::{collections::HashMap, sync::Arc};

// Simple test context for instruction metering
#[derive(Debug, Clone, Default)]
struct SimpleContext {
    remaining: u64,
}

impl solana_sbpf::vm::ContextObject for SimpleContext {
    fn consume(&mut self, amount: u64) {
        self.remaining = self.remaining.saturating_sub(amount);
    }
    fn get_remaining(&self) -> u64 {
        self.remaining
    }
}

/// A simple "Solana-like" program that demonstrates various operations:
/// - Function calls
/// - Memory operations
/// - Arithmetic
/// - Conditional branching
const SAMPLE_PROGRAM: &str = r#"
    mov64 r1, 100
    mov64 r2, 50
    call process_data
    stxdw [r10-8], r0
    ldxdw r1, [r10-8]
    jeq r1, r0, +1
    mov64 r0, 0
    exit
process_data:
    stxdw [r10-16], r6
    stxdw [r10-24], r7
    add64 r1, r2
    mov64 r6, r1
    call double_value
    ldxdw r6, [r10-16]
    ldxdw r7, [r10-24]
    exit
double_value:
    mov64 r0, r1
    add64 r0, r1
    exit
"#;

/// A simpler program for basic tracing demonstration
const SIMPLE_PROGRAM: &str = r#"
    mov64 r0, 0
    mov64 r1, 5
loop_start:
    add64 r0, 1
    sub64 r1, 1
    jne r1, 0, -3
    exit
"#;

/// Program demonstrating memory operations
const MEMORY_PROGRAM: &str = r#"
    mov64 r1, 0x1234
    stxdw [r10-8], r1
    mov64 r2, 0x5678
    stxdw [r10-16], r2
    ldxdw r3, [r10-8]
    ldxdw r4, [r10-16]
    xor64 r3, r4
    mov64 r0, r3
    exit
"#;

fn create_memory_mapping<'a>(
    executable: &'a Executable<SimpleContext>,
    stack: &'a mut solana_sbpf::aligned_memory::AlignedMemory<{ ebpf::HOST_ALIGN }>,
    heap: &'a mut solana_sbpf::aligned_memory::AlignedMemory<{ ebpf::HOST_ALIGN }>,
) -> solana_sbpf::memory_region::MemoryMapping<'a> {
    use solana_sbpf::memory_region::MemoryRegion;

    let config = executable.get_config();
    let sbpf_version = executable.get_sbpf_version();

    // Use simple writable region for stack (no gaps for simplicity)
    let regions: Vec<MemoryRegion> = vec![
        executable.get_ro_region(),
        MemoryRegion::new_writable(stack.as_slice_mut(), ebpf::MM_STACK_START),
        MemoryRegion::new_writable(heap.as_slice_mut(), ebpf::MM_HEAP_START),
    ];

    solana_sbpf::memory_region::MemoryMapping::new(regions, config, sbpf_version).unwrap()
}

fn run_traced_program(
    name: &str,
    source: &str,
    symbols: HashMap<u64, String>,
    enable_cfg: bool,
    enable_dataflow: bool,
) {
    println!("\n{}", "=".repeat(60));
    println!("  Running: {}", name);
    println!("{}\n", "=".repeat(60));

    let config = Config {
        enable_instruction_meter: true,
        enable_stack_frame_gaps: false,
        ..Config::default()
    };

    let loader = Arc::new(BuiltinProgram::new_loader(config.clone()));

    // Assemble the program
    let executable = match assemble::<SimpleContext>(source, loader.clone()) {
        Ok(exe) => exe,
        Err(e) => {
            eprintln!("Assembly error: {:?}", e);
            return;
        }
    };

    // Verify the program
    if let Err(e) = executable.verify::<RequisiteVerifier>() {
        eprintln!("Verification error: {:?}", e);
        return;
    }

    // Create VM components
    let mut stack = solana_sbpf::aligned_memory::AlignedMemory::zero_filled(config.stack_size());
    let mut heap = solana_sbpf::aligned_memory::AlignedMemory::with_capacity(0);
    let stack_len = stack.len();

    let memory_mapping = create_memory_mapping(&executable, &mut stack, &mut heap);

    let mut context = SimpleContext { remaining: 10000 };

    let mut vm = EbpfVm::new(
        loader,
        executable.get_sbpf_version(),
        &mut context,
        memory_mapping,
        stack_len,
    );

    // Set up input register (r1 typically points to input data in Solana)
    vm.registers[1] = ebpf::MM_INPUT_START;

    // Create tracer with symbols
    let mut tracer = if symbols.is_empty() {
        Tracer::from_executable(&executable, Some(name.to_string()))
    } else {
        Tracer::new(symbols, Some(name.to_string()))
    };

    // Enable CFG analysis if requested
    if enable_cfg {
        tracer.enable_cfg_analysis();
        println!("CFG analysis enabled");
    }

    // Enable dataflow analysis if requested
    if enable_dataflow {
        tracer.enable_dataflow(true);
        println!("Data flow analysis enabled (full tracking)");
    }

    // Execute with tracing
    println!("\nExecuting program...\n");
    let (insn_count, result, trace) = tracer.execute(&mut vm, &executable);

    // Print summary
    println!("Execution Summary:");
    println!("  Instructions executed: {}", insn_count);
    println!("  Result: {:?}", result);
    println!("  Total CUs: {}", trace.total_compute_units);

    // Print trace structure summary
    println!("\nTrace Structure:");
    print_frame_summary(&trace.execution_tree, 0);

    // Print CFG summary if available
    if let Some(cfg) = &trace.control_flow_graph {
        println!("\n--- Control Flow Graph Analysis ---");
        println!("  Basic blocks: {}", cfg.basic_blocks.len());
        println!("  Edges: {}", cfg.edges.len());
        println!("  Entry PC: {}", cfg.entry_pc);
        println!("  Exit PCs: {:?}", cfg.exit_pcs);
        println!("  Loops detected: {}", cfg.loops.len());

        // Print basic blocks
        println!("\n  Basic Blocks:");
        for (pc, block) in &cfg.basic_blocks {
            println!(
                "    PC {}: {} instructions, {:?}, label={:?}, exec_count={}",
                pc,
                block.instruction_count,
                block.terminator,
                block.label,
                block.execution_count
            );
        }

        // Print loop info
        if !cfg.loops.is_empty() {
            println!("\n  Detected Loops:");
            for (i, loop_info) in cfg.loops.iter().enumerate() {
                println!(
                    "    Loop {}: header=PC {}, iterations={}, back_edge=PC {}",
                    i, loop_info.header_pc, loop_info.iteration_count, loop_info.back_edge_pc
                );
                if let Some(bound) = &loop_info.estimated_bound {
                    println!("      Estimated bound: {:?}", bound);
                }
            }
        }
    }

    // Print dataflow summary if available
    if let Some(dataflow) = &trace.dataflow {
        println!("\n--- Data Flow Analysis ---");
        println!("  Value definitions: {}", dataflow.definitions.len());
        println!("  Value uses: {}", dataflow.uses.len());
        println!("  Def-use chains: {}", dataflow.def_use_chains.len());
        println!("  Memory stores tracked: {}", dataflow.memory_stores.len());

        // Print some interesting definitions
        println!("\n  Sample Definitions (first 5):");
        for (def_id, def) in dataflow.definitions.iter().take(5) {
            println!(
                "    {}: {:?} -> {:?}",
                def_id, def.origin, def.destination
            );
            if !def.taint.is_empty() {
                println!("      Taint: {:?}", def.taint);
            }
        }
    }

    // Print full JSON trace
    println!("\n--- Full JSON Trace ---\n");
    match Tracer::to_json(&trace) {
        Ok(json) => println!("{}", json),
        Err(e) => eprintln!("JSON serialization error: {:?}", e),
    }
}

fn print_frame_summary(frames: &[solana_sbpf::tracer::TracedCallFrame], indent: usize) {
    for frame in frames {
        let prefix = "  ".repeat(indent);
        let name = frame.symbol_name.as_deref().unwrap_or("<unknown>");
        let event_count = frame.events.len();
        let call_count = frame.sub_calls.len();

        println!(
            "{}[depth={}] {} - {} events, {} sub-calls",
            prefix, frame.depth, name, event_count, call_count
        );

        // Count event types
        let mut insn_count = 0;
        let mut mem_count = 0;
        let mut syscall_count = 0;
        let mut call_count = 0;
        let mut return_count = 0;

        for event in &frame.events {
            match event {
                solana_sbpf::tracer::TraceEvent::Instruction(_) => insn_count += 1,
                solana_sbpf::tracer::TraceEvent::MemoryAccess(_) => mem_count += 1,
                solana_sbpf::tracer::TraceEvent::Syscall(_) => syscall_count += 1,
                solana_sbpf::tracer::TraceEvent::FunctionCall(_) => call_count += 1,
                solana_sbpf::tracer::TraceEvent::FunctionReturn(_) => return_count += 1,
            }
        }

        println!(
            "{}  Events: {} insns, {} mem, {} syscalls, {} calls, {} returns",
            prefix, insn_count, mem_count, syscall_count, call_count, return_count
        );

        // Recurse into sub-calls
        if !frame.sub_calls.is_empty() {
            print_frame_summary(&frame.sub_calls, indent + 1);
        }
    }
}

fn main() {
    println!("Solana SBPF Semantic Tracer Demo");
    println!("================================");
    println!("Demonstrating: Execution tracing, CFG analysis, Data flow analysis\n");

    // Run simple counter program with CFG analysis to show loop detection
    println!("Example 1: Loop detection with CFG analysis");
    run_traced_program(
        "Simple Counter (Loop)",
        SIMPLE_PROGRAM,
        HashMap::new(),
        true,  // Enable CFG
        false, // Skip dataflow for this one
    );

    // Run memory operations program with dataflow to show value tracking
    println!("\n\nExample 2: Memory operations with data flow tracking");
    run_traced_program(
        "Memory Operations",
        MEMORY_PROGRAM,
        HashMap::new(),
        false, // Skip CFG for this one
        true,  // Enable dataflow
    );

    // Run complex program with both analyses and custom symbols
    println!("\n\nExample 3: Full analysis with function calls");
    let mut symbols = HashMap::new();
    symbols.insert(0, "entrypoint".to_string());
    symbols.insert(8, "process_data".to_string());
    symbols.insert(16, "double_value".to_string());

    run_traced_program(
        "Complex Function Calls",
        SAMPLE_PROGRAM,
        symbols,
        true, // Enable CFG
        true, // Enable dataflow
    );

    println!("\n\nDemo complete!");
    println!("The JSON output can be used by LLMs for program analysis and decompilation.");
}
