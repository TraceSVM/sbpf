//! Integration tests for semantic execution tracing.

#![cfg(feature = "semantic-tracer")]

use solana_sbpf::{
    assembler::assemble,
    error::ProgramResult,
    program::BuiltinProgram,
    tracer::{
        cfg::{BlockTerminator, BranchCondition, EdgeType},
        dataflow::{OperationType, TaintLabel, ValueOrigin},
        MemoryRegionType, TraceContext, TraceEvent, TraceResult, Tracer,
    },
    verifier::RequisiteVerifier,
    vm::Config,
};
use std::sync::Arc;
use test_utils::{create_vm, TestContextObject};

/// Helper to assemble and execute a traced program.
fn execute_traced(source: &str, remaining: u64) -> (u64, ProgramResult, TraceContext) {
    let config = Config {
        enable_instruction_meter: true,
        ..Config::default()
    };
    let loader = Arc::new(BuiltinProgram::new_loader(config));
    let executable = assemble::<TestContextObject>(source, loader).unwrap();
    executable.verify::<RequisiteVerifier>().unwrap();

    let mut context_object = TestContextObject::new(remaining);
    create_vm!(
        vm,
        &executable,
        &mut context_object,
        stack,
        heap,
        Vec::new(),
        None
    );

    let mut tracer = Tracer::from_executable(&executable, Some("test_program".to_string()));
    tracer.execute(&mut vm, &executable)
}

#[test]
fn test_simple_mov_exit() {
    let source = "
        mov64 r0, 42
        exit
    ";

    let (insn_count, result, trace) = execute_traced(source, 100);

    // Check execution succeeded
    assert!(matches!(result, ProgramResult::Ok(42)));
    assert!(insn_count > 0);

    // Check trace structure
    assert_eq!(trace.program_id, Some("test_program".to_string()));
    assert!(matches!(trace.result, TraceResult::Success { return_value: 42 }));
    assert!(!trace.execution_tree.is_empty());

    // Check we have instruction events
    let root_frame = &trace.execution_tree[0];
    assert!(root_frame.events.len() >= 2); // At least mov64 and exit

    // First event should be mov64 instruction
    if let TraceEvent::Instruction(insn) = &root_frame.events[0] {
        assert!(insn.mnemonic.contains("mov64"));
        assert_eq!(insn.pc, 0);
    } else {
        panic!("Expected instruction event");
    }
}

#[test]
fn test_arithmetic_operations() {
    let source = "
        mov64 r1, 10
        mov64 r2, 5
        add64 r1, r2
        mov64 r0, r1
        exit
    ";

    let (_, result, trace) = execute_traced(source, 100);

    assert!(matches!(result, ProgramResult::Ok(15)));

    let root_frame = &trace.execution_tree[0];

    // Find the add64 instruction
    let has_add = root_frame.events.iter().any(|e| {
        if let TraceEvent::Instruction(insn) = e {
            insn.mnemonic.contains("add64")
        } else {
            false
        }
    });
    assert!(has_add, "Should have add64 instruction in trace");
}

#[test]
fn test_register_diff_tracking() {
    let source = "
        mov64 r0, 0
        mov64 r1, 100
        mov64 r2, 200
        exit
    ";

    let (_, _, trace) = execute_traced(source, 100);

    let root_frame = &trace.execution_tree[0];

    // Find mov64 r1, 100 instruction and check its register changes
    let has_r1_change = root_frame.events.iter().any(|e| {
        if let TraceEvent::Instruction(insn) = e {
            insn.register_changes.iter().any(|rc| rc.register == 1 && rc.value_after == 100)
        } else {
            false
        }
    });
    assert!(has_r1_change, "Should track r1 change to 100");
}

#[test]
fn test_memory_classification() {
    use solana_sbpf::tracer::classify_pointer;
    use solana_sbpf::ebpf::{MM_STACK_START, MM_HEAP_START, MM_INPUT_START, MM_BYTECODE_START};

    assert_eq!(classify_pointer(0), MemoryRegionType::Rodata);
    assert_eq!(classify_pointer(MM_BYTECODE_START), MemoryRegionType::Bytecode);
    assert_eq!(classify_pointer(MM_STACK_START), MemoryRegionType::Stack);
    assert_eq!(classify_pointer(MM_HEAP_START), MemoryRegionType::Heap);
    assert_eq!(classify_pointer(MM_INPUT_START), MemoryRegionType::Input);
}

#[test]
fn test_json_serialization() {
    let source = "
        mov64 r0, 1
        exit
    ";

    let (_, _, trace) = execute_traced(source, 100);

    // Test JSON serialization doesn't panic
    let json = Tracer::to_json(&trace).unwrap();
    assert!(json.contains("test_program"));
    assert!(json.contains("execution_tree"));
    assert!(json.contains("mov64"));

    // Test compact JSON
    let compact_json = Tracer::to_json_compact(&trace).unwrap();
    assert!(compact_json.len() < json.len());
}

#[test]
fn test_compute_units_tracking() {
    let source = "
        mov64 r0, 0
        add64 r0, 1
        add64 r0, 1
        add64 r0, 1
        exit
    ";

    let (insn_count, _, trace) = execute_traced(source, 100);

    // Should have consumed some CUs (5 instructions)
    assert!(insn_count >= 5, "insn_count = {}", insn_count);
    assert!(trace.total_compute_units >= 5, "total_compute_units = {}", trace.total_compute_units);

    // Verify trace has the expected number of instruction events
    let root_frame = &trace.execution_tree[0];
    let instruction_count = root_frame.events.iter().filter(|e| {
        matches!(e, TraceEvent::Instruction(_))
    }).count();
    assert!(instruction_count >= 5, "Should have at least 5 instruction events, got {}", instruction_count);
}

#[test]
fn test_tracer_symbol_map() {
    use std::collections::HashMap;

    let mut symbol_map = HashMap::new();
    symbol_map.insert(0u64, "main".to_string());
    symbol_map.insert(10u64, "helper".to_string());

    // Create tracer with symbol map - test that it can be created
    let _tracer = Tracer::new(symbol_map, Some("test".to_string()));
    // If this doesn't panic, the test passes
}

#[test]
fn test_conditional_branch() {
    let source = "
        mov64 r0, 0
        mov64 r1, 5
        jeq r1, 5, +1
        mov64 r0, 100
        mov64 r0, 42
        exit
    ";

    let (_, result, trace) = execute_traced(source, 100);

    // Should skip mov64 r0, 100 and execute mov64 r0, 42
    assert!(matches!(result, ProgramResult::Ok(42)));

    // Check that trace has the branch instruction
    let root_frame = &trace.execution_tree[0];
    let has_jeq = root_frame.events.iter().any(|e| {
        if let TraceEvent::Instruction(insn) = e {
            insn.mnemonic.contains("jeq")
        } else {
            false
        }
    });
    assert!(has_jeq, "Should have conditional branch in trace");
}

#[test]
fn test_loop_tracing() {
    let source = "
        mov64 r0, 0
        mov64 r1, 3
    loop:
        add64 r0, 1
        sub64 r1, 1
        jne r1, 0, -3
        exit
    ";

    let (_, result, trace) = execute_traced(source, 100);

    // Should loop 3 times, r0 = 3
    assert!(matches!(result, ProgramResult::Ok(3)));

    // Should have multiple add64 instructions in trace (from loop iterations)
    let root_frame = &trace.execution_tree[0];
    let add_count = root_frame.events.iter().filter(|e| {
        if let TraceEvent::Instruction(insn) = e {
            insn.mnemonic.contains("add64")
        } else {
            false
        }
    }).count();

    assert_eq!(add_count, 3, "Should have 3 add64 instructions from loop");
}

#[test]
fn test_stack_memory_access() {
    // Test that we can trace stack memory operations
    let source = "
        mov64 r0, 42
        stxdw [r10-8], r0
        ldxdw r0, [r10-8]
        exit
    ";

    let (_, result, trace) = execute_traced(source, 100);

    assert!(matches!(result, ProgramResult::Ok(42)));

    // Should have memory access events
    let root_frame = &trace.execution_tree[0];
    let has_memory_event = root_frame.events.iter().any(|e| {
        matches!(e, TraceEvent::MemoryAccess(_))
    });
    assert!(has_memory_event, "Should have memory access events");
}

#[test]
fn test_trace_result_error() {
    // Test with insufficient CUs to trigger error
    let source = "
        mov64 r0, 0
        add64 r0, 1
        add64 r0, 1
        add64 r0, 1
        add64 r0, 1
        add64 r0, 1
        exit
    ";

    let (_, _, trace) = execute_traced(source, 2); // Only 2 CUs

    // Should have an error result due to exceeded instruction meter
    assert!(matches!(trace.result, TraceResult::Error { .. }));
}

#[test]
fn test_hex_formatting() {
    use solana_sbpf::tracer::format_hex_value;

    let (hex, truncated) = format_hex_value(&[0xde, 0xad, 0xbe, 0xef], 64);
    assert_eq!(hex, "deadbeef");
    assert!(!truncated);

    // Test truncation
    let large = vec![0xab; 100];
    let (hex, truncated) = format_hex_value(&large, 64);
    assert!(truncated);
    assert_eq!(hex.len(), 128); // 64 bytes * 2 hex chars
}

#[test]
fn test_syscall_decoder_registry() {
    use solana_sbpf::tracer::SyscallDecoderRegistry;
    use solana_sbpf::memory_region::MemoryMapping;

    let registry = SyscallDecoderRegistry::new();

    // Test fallback decode for unknown syscall
    let args = [1u64, 2, 3, 4, 5];
    let decoded = registry.decode("unknown_syscall", args, &MemoryMapping::Identity);

    assert_eq!(decoded.get("r1"), Some(&"0x1".to_string()));
    assert_eq!(decoded.get("r5"), Some(&"0x5".to_string()));
}

#[test]
fn test_function_call_tracing() {
    // Test internal function call tracing
    let source = "
        mov64 r1, 10
        mov64 r2, 20
        call my_add
        exit
        my_add:
        add64 r1, r2
        mov64 r0, r1
        exit
    ";

    let (_, result, trace) = execute_traced(source, 100);

    // Should return 30 (10 + 20)
    assert!(matches!(result, ProgramResult::Ok(30)));

    // Check for function call event in the root frame
    let root_frame = &trace.execution_tree[0];

    let call_events: Vec<_> = root_frame.events.iter().filter_map(|e| {
        if let TraceEvent::FunctionCall(call) = e {
            Some(call)
        } else {
            None
        }
    }).collect();

    assert!(!call_events.is_empty(), "Should have at least one FunctionCall event");

    // Verify call event structure (function name may or may not be resolved
    // depending on how the executable was built)
    let call = &call_events[0];
    assert!(!call.is_indirect, "Should be a direct call, not indirect");
    assert_eq!(call.args[0], 10, "First argument should be 10");
    assert_eq!(call.args[1], 20, "Second argument should be 20");
    assert!(call.target_pc > 0, "Should have valid target PC");

    // Check for function return event
    let return_events: Vec<_> = root_frame.events.iter().filter_map(|e| {
        if let TraceEvent::FunctionReturn(ret) = e {
            Some(ret)
        } else {
            None
        }
    }).collect();

    assert!(!return_events.is_empty(), "Should have at least one FunctionReturn event");
    assert_eq!(return_events[0].return_value, 30, "Return value should be 30");

    // Check sub_calls contains the called function's frame
    assert!(!root_frame.sub_calls.is_empty(), "Should have sub_calls");

    // Verify JSON output contains FunctionCall and FunctionReturn
    let json = Tracer::to_json(&trace).unwrap();
    assert!(json.contains("FunctionCall"), "JSON should contain FunctionCall events");
    assert!(json.contains("FunctionReturn"), "JSON should contain FunctionReturn events");
    assert!(json.contains("target_pc"), "JSON should contain target_pc");
    assert!(json.contains("call_site_pc"), "JSON should contain call_site_pc");
}

#[test]
fn test_nested_function_calls() {
    // Test nested function call tracing - verify hierarchical structure
    let source = "
        mov64 r1, 5
        call outer_fn
        exit
        outer_fn:
        mov64 r2, 10
        call inner_fn
        mov64 r0, r1
        exit
        inner_fn:
        add64 r1, r2
        exit
    ";

    let (_, result, trace) = execute_traced(source, 100);

    // Should return 15 (5 + 10)
    assert!(matches!(result, ProgramResult::Ok(15)));

    // Root frame should have one sub_call (the outer function)
    let root_frame = &trace.execution_tree[0];
    assert!(!root_frame.sub_calls.is_empty(), "Root should have sub_calls");

    // Verify nested structure: root -> outer -> inner
    let outer_frame = &root_frame.sub_calls[0];
    assert!(outer_frame.depth == 1, "outer_fn should be at depth 1");

    // outer frame should have one sub_call (inner function)
    assert!(!outer_frame.sub_calls.is_empty(), "outer frame should have sub_calls");
    let inner_frame = &outer_frame.sub_calls[0];
    assert!(inner_frame.depth == 2, "inner_fn should be at depth 2");

    // Verify the hierarchy is properly nested
    assert_eq!(root_frame.sub_calls.len(), 1, "Root should have exactly 1 sub_call");
    assert_eq!(outer_frame.sub_calls.len(), 1, "Outer should have exactly 1 sub_call");
    assert!(inner_frame.sub_calls.is_empty(), "Inner should have no sub_calls");

    // Verify JSON output structure
    let json = Tracer::to_json(&trace).unwrap();
    assert!(json.contains("FunctionCall"), "JSON should contain FunctionCall events");
    assert!(json.contains("FunctionReturn"), "JSON should contain FunctionReturn events");
    assert!(json.contains("sub_calls"), "JSON should contain sub_calls");
}

#[test]
fn test_function_call_with_custom_symbols() {
    use std::collections::HashMap;

    // Test that manually adding symbols works
    let source = "
        mov64 r1, 42
        call helper
        exit
        helper:
        mov64 r0, r1
        exit
    ";

    let config = Config {
        enable_instruction_meter: true,
        ..Config::default()
    };
    let loader = Arc::new(BuiltinProgram::new_loader(config));
    let executable = assemble::<TestContextObject>(source, loader).unwrap();
    executable.verify::<RequisiteVerifier>().unwrap();

    let mut context_object = TestContextObject::new(100);
    create_vm!(
        vm,
        &executable,
        &mut context_object,
        stack,
        heap,
        Vec::new(),
        None
    );

    // Create tracer with custom symbol map
    // PC layout: 0=mov64, 1=call, 2=exit, 3=helper:mov64, 4=exit
    let mut symbol_map = HashMap::new();
    symbol_map.insert(3u64, "my_helper_function".to_string()); // PC 3 is where helper starts

    let mut tracer = Tracer::new(symbol_map, Some("test_program".to_string()));
    let (_, result, trace) = tracer.execute(&mut vm, &executable);

    assert!(matches!(result, ProgramResult::Ok(42)));

    // Verify the custom symbol name appears in the trace
    let json = Tracer::to_json(&trace).unwrap();
    assert!(json.contains("my_helper_function"), "JSON should contain custom symbol name");
}

/// Helper to assemble and execute a traced program with CFG and dataflow enabled.
fn execute_traced_with_analysis(source: &str, remaining: u64) -> (u64, ProgramResult, TraceContext) {
    let config = Config {
        enable_instruction_meter: true,
        ..Config::default()
    };
    let loader = Arc::new(BuiltinProgram::new_loader(config));
    let executable = assemble::<TestContextObject>(source, loader).unwrap();
    executable.verify::<RequisiteVerifier>().unwrap();

    let mut context_object = TestContextObject::new(remaining);
    create_vm!(
        vm,
        &executable,
        &mut context_object,
        stack,
        heap,
        Vec::new(),
        None
    );

    let mut tracer = Tracer::from_executable(&executable, Some("test_program".to_string()));
    tracer.enable_cfg_analysis();
    tracer.enable_dataflow(true);
    tracer.execute(&mut vm, &executable)
}

// ============================================================================
// CFG Analysis Tests
// ============================================================================

#[test]
fn test_cfg_basic_blocks() {
    let source = "
        mov64 r0, 42
        exit
    ";

    let (_, result, trace) = execute_traced_with_analysis(source, 100);

    assert!(matches!(result, ProgramResult::Ok(42)));

    // Should have CFG data
    let cfg = trace.control_flow_graph.as_ref().expect("Should have CFG");

    // Should have at least one basic block
    assert!(!cfg.basic_blocks.is_empty(), "Should have basic blocks");

    // Entry block should start at PC 0
    assert_eq!(cfg.entry_pc, 0, "Entry PC should be 0");

    // Should have exit PCs
    assert!(!cfg.exit_pcs.is_empty(), "Should have exit PCs");
}

#[test]
fn test_cfg_conditional_branch() {
    let source = "
        mov64 r1, 5
        jeq r1, 5, +1
        mov64 r0, 0
        mov64 r0, 42
        exit
    ";

    let (_, result, trace) = execute_traced_with_analysis(source, 100);

    assert!(matches!(result, ProgramResult::Ok(42)));

    let cfg = trace.control_flow_graph.as_ref().expect("Should have CFG");

    // Should have multiple basic blocks due to conditional branch
    assert!(cfg.basic_blocks.len() >= 2, "Should have multiple basic blocks");

    // Should have edges
    assert!(!cfg.edges.is_empty(), "Should have edges");

    // Find the block with conditional terminator
    let has_conditional = cfg.basic_blocks.values().any(|b| {
        matches!(b.terminator, BlockTerminator::ConditionalBranch { .. })
    });
    assert!(has_conditional, "Should have a block with conditional branch terminator");

    // Should have conditional taken edge
    let has_conditional_edge = cfg.edges.iter().any(|e| {
        matches!(e.edge_type, EdgeType::ConditionalTaken | EdgeType::ConditionalFallthrough)
    });
    assert!(has_conditional_edge, "Should have conditional edge types");
}

#[test]
fn test_cfg_loop_detection() {
    let source = "
        mov64 r0, 0
        mov64 r1, 3
    loop:
        add64 r0, 1
        sub64 r1, 1
        jne r1, 0, -3
        exit
    ";

    let (_, result, trace) = execute_traced_with_analysis(source, 100);

    assert!(matches!(result, ProgramResult::Ok(3)));

    let cfg = trace.control_flow_graph.as_ref().expect("Should have CFG");

    // Should detect at least one loop
    assert!(!cfg.loops.is_empty(), "Should detect loops");

    // Loop should have multiple iterations
    let loop_info = &cfg.loops[0];
    assert!(loop_info.iteration_count >= 3, "Loop should have at least 3 iterations");

    // Should have a back edge
    let has_back_edge = cfg.edges.iter().any(|e| {
        matches!(e.edge_type, EdgeType::BackEdge)
    });
    assert!(has_back_edge, "Should have back edge for loop");
}

#[test]
fn test_cfg_branch_condition_inference() {
    let source = "
        mov64 r1, 10
        mov64 r2, 5
        jgt r1, r2, +1
        mov64 r0, 0
        mov64 r0, 1
        exit
    ";

    let (_, result, trace) = execute_traced_with_analysis(source, 100);

    assert!(matches!(result, ProgramResult::Ok(1))); // Branch taken

    let cfg = trace.control_flow_graph.as_ref().expect("Should have CFG");

    // Find block with conditional branch and verify condition
    let conditional_block = cfg.basic_blocks.values().find(|b| {
        matches!(b.terminator, BlockTerminator::ConditionalBranch { .. })
    });

    assert!(conditional_block.is_some(), "Should have conditional branch block");

    if let Some(block) = conditional_block {
        if let BlockTerminator::ConditionalBranch { condition, .. } = &block.terminator {
            assert!(matches!(condition, BranchCondition::GreaterThan), "Should be greater than condition");
        }
    }
}

#[test]
fn test_cfg_function_call_edges() {
    let source = "
        mov64 r1, 10
        call helper
        exit
        helper:
        mov64 r0, r1
        exit
    ";

    let (_, result, trace) = execute_traced_with_analysis(source, 100);

    assert!(matches!(result, ProgramResult::Ok(10)));

    let cfg = trace.control_flow_graph.as_ref().expect("Should have CFG");

    // Should have call and return edges
    let has_call_edge = cfg.edges.iter().any(|e| {
        matches!(e.edge_type, EdgeType::Call)
    });
    assert!(has_call_edge, "Should have call edge");

    // Should have function call terminator
    let has_call_terminator = cfg.basic_blocks.values().any(|b| {
        matches!(b.terminator, BlockTerminator::FunctionCall { .. })
    });
    assert!(has_call_terminator, "Should have function call terminator");
}

// ============================================================================
// Data Flow Analysis Tests
// ============================================================================

#[test]
fn test_dataflow_constant_propagation() {
    let source = "
        mov64 r0, 42
        exit
    ";

    let (_, result, trace) = execute_traced_with_analysis(source, 100);

    assert!(matches!(result, ProgramResult::Ok(42)));

    let dataflow = trace.dataflow.as_ref().expect("Should have dataflow");

    // Should have definitions
    assert!(!dataflow.definitions.is_empty(), "Should have definitions");

    // Find a constant origin definition
    let has_constant = dataflow.definitions.values().any(|def| {
        matches!(def.origin, ValueOrigin::Constant { value: 42 })
    });
    assert!(has_constant, "Should have constant value definition for 42");
}

#[test]
fn test_dataflow_register_copy() {
    let source = "
        mov64 r1, 100
        mov64 r0, r1
        exit
    ";

    let (_, result, trace) = execute_traced_with_analysis(source, 100);

    assert!(matches!(result, ProgramResult::Ok(100)));

    let dataflow = trace.dataflow.as_ref().expect("Should have dataflow");

    // Should have register copy origin
    let has_copy = dataflow.definitions.values().any(|def| {
        matches!(def.origin, ValueOrigin::RegisterCopy { source_reg: 1, .. })
    });
    assert!(has_copy, "Should have register copy definition");
}

#[test]
fn test_dataflow_computed_values() {
    let source = "
        mov64 r1, 10
        mov64 r2, 5
        add64 r1, r2
        mov64 r0, r1
        exit
    ";

    let (_, result, trace) = execute_traced_with_analysis(source, 100);

    assert!(matches!(result, ProgramResult::Ok(15)));

    let dataflow = trace.dataflow.as_ref().expect("Should have dataflow");

    // Should have computed value origin (add operation)
    let has_computed = dataflow.definitions.values().any(|def| {
        if let ValueOrigin::Computed { operation, .. } = &def.origin {
            matches!(operation, OperationType::Add)
        } else {
            false
        }
    });
    assert!(has_computed, "Should have computed add definition");
}

#[test]
fn test_dataflow_taint_tracking() {
    let source = "
        mov64 r0, 42
        exit
    ";

    let (_, result, trace) = execute_traced_with_analysis(source, 100);

    assert!(matches!(result, ProgramResult::Ok(42)));

    let dataflow = trace.dataflow.as_ref().expect("Should have dataflow");

    // Constant values should be tainted as Constant
    let has_constant_taint = dataflow.definitions.values().any(|def| {
        def.taint.iter().any(|t| matches!(t, TaintLabel::Constant))
    });
    assert!(has_constant_taint, "Constant should have Constant taint label");
}

#[test]
fn test_dataflow_entry_parameters() {
    let source = "
        mov64 r0, r1
        exit
    ";

    let (_, _, trace) = execute_traced_with_analysis(source, 100);

    let dataflow = trace.dataflow.as_ref().expect("Should have dataflow");

    // Should have entry parameter origins for initial r1-r5
    let has_entry_param = dataflow.definitions.values().any(|def| {
        matches!(def.origin, ValueOrigin::EntryParameter { .. })
    });
    assert!(has_entry_param, "Should have entry parameter definitions");

    // Entry parameters should have InputArg taint
    let has_input_taint = dataflow.definitions.values().any(|def| {
        def.taint.iter().any(|t| matches!(t, TaintLabel::InputArg { .. }))
    });
    assert!(has_input_taint, "Entry parameters should have InputArg taint");
}

#[test]
fn test_dataflow_def_use_chains() {
    let source = "
        mov64 r1, 10
        mov64 r2, r1
        mov64 r0, r2
        exit
    ";

    let (_, result, trace) = execute_traced_with_analysis(source, 100);

    assert!(matches!(result, ProgramResult::Ok(10)));

    let dataflow = trace.dataflow.as_ref().expect("Should have dataflow");

    // Should have uses
    assert!(!dataflow.uses.is_empty(), "Should have value uses");

    // Should have def-use chains
    assert!(!dataflow.def_use_chains.is_empty(), "Should have def-use chains");
}

#[test]
fn test_dataflow_memory_store_tracking() {
    let source = "
        mov64 r1, 42
        stxdw [r10-8], r1
        ldxdw r0, [r10-8]
        exit
    ";

    let (_, result, trace) = execute_traced_with_analysis(source, 100);

    assert!(matches!(result, ProgramResult::Ok(42)));

    let dataflow = trace.dataflow.as_ref().expect("Should have dataflow");

    // Should have memory stores tracked
    assert!(!dataflow.memory_stores.is_empty(), "Should track memory stores");

    // Should have memory load origin that references the store
    let has_memory_load = dataflow.definitions.values().any(|def| {
        matches!(def.origin, ValueOrigin::MemoryLoad { .. })
    });
    assert!(has_memory_load, "Should have memory load definition");
}

#[test]
fn test_dataflow_taint_propagation_through_operations() {
    let source = "
        mov64 r1, 10
        add64 r1, 5
        mov64 r0, r1
        exit
    ";

    let (_, result, trace) = execute_traced_with_analysis(source, 100);

    assert!(matches!(result, ProgramResult::Ok(15)));

    let dataflow = trace.dataflow.as_ref().expect("Should have dataflow");

    // The computed value should have Constant taint (both operands are constants)
    let computed_def = dataflow.definitions.values().find(|def| {
        matches!(def.origin, ValueOrigin::Computed { .. })
    });

    assert!(computed_def.is_some(), "Should have computed definition");
    let def = computed_def.unwrap();
    assert!(def.taint.iter().any(|t| matches!(t, TaintLabel::Constant)),
        "Computed value from constants should have Constant taint");
}

// ============================================================================
// Combined CFG + Dataflow Tests
// ============================================================================

#[test]
fn test_combined_analysis_json_output() {
    let source = "
        mov64 r0, 0
        mov64 r1, 3
    loop:
        add64 r0, 1
        sub64 r1, 1
        jne r1, 0, -3
        exit
    ";

    let (_, result, trace) = execute_traced_with_analysis(source, 100);

    assert!(matches!(result, ProgramResult::Ok(3)));

    // Verify both analyses are present
    assert!(trace.control_flow_graph.is_some(), "Should have CFG");
    assert!(trace.dataflow.is_some(), "Should have dataflow");

    // Test JSON output contains both
    let json = Tracer::to_json(&trace).unwrap();
    assert!(json.contains("control_flow_graph"), "JSON should contain CFG");
    assert!(json.contains("basic_blocks"), "JSON should contain basic blocks");
    assert!(json.contains("dataflow"), "JSON should contain dataflow");
    assert!(json.contains("definitions"), "JSON should contain definitions");
    assert!(json.contains("loops"), "JSON should contain loop info");
}
