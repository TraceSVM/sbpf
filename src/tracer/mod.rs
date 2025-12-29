//! Semantic execution tracing for SBPF programs.
//!
//! This module provides a tracer that generates hierarchical, JSON-serializable
//! execution traces suitable for LLM analysis of Solana programs.
//!
//! # Example
//!
//! ```ignore
//! use solana_sbpf::tracer::Tracer;
//!
//! let mut tracer = Tracer::new(HashMap::new(), Some("program_id".to_string()));
//! let (insn_count, result, trace) = tracer.execute(&mut vm, &executable);
//! let json = Tracer::to_json(&trace).unwrap();
//! ```

use crate::{
    disassembler::disassemble_instruction,
    ebpf::{self, Insn},
    elf::Executable,
    error::ProgramResult,
    interpreter::Interpreter,
    memory_region::{AccessType, MemoryMapping},
    vm::{ContextObject, EbpfVm},
};
use std::collections::{BTreeMap, HashMap};

pub mod cfg;
pub mod dataflow;
pub mod memory;
pub mod queryable;
pub mod syscall_decoder;
pub mod types;

pub use cfg::{
    BasicBlock, BlockTerminator, BranchCondition, BranchComparand, BranchTakenInfo, CfgBuilder,
    CfgEdge, ControlFlowGraph, DetectedLoop, EdgeType, LoopBound,
};
pub use dataflow::{
    DataFlowAnalyzer, DataFlowState, DefId, MemoryStore, OperationType, TaintLabel,
    UseType, ValueDefinition, ValueLocation, ValueOrigin, ValueUse,
};
pub use memory::{classify_pointer, format_hex_value, format_hex_value_default};
pub use queryable::{QueryableTrace, ExecutionSummary, FunctionIndexEntry, FunctionTrace, SCHEMA_DESCRIPTION};
pub use syscall_decoder::{SyscallDecoder, SyscallDecoderRegistry};
pub use types::*;

/// Semantic tracer that wraps VM execution.
///
/// The tracer captures instruction execution, memory accesses, syscalls,
/// and function call frames in a hierarchical structure.
pub struct Tracer {
    /// Symbol map: PC -> function name.
    symbol_map: HashMap<u64, String>,
    /// Syscall decoder registry.
    decoder_registry: SyscallDecoderRegistry,
    /// Current frame stack for hierarchical tracing.
    frame_stack: Vec<TracedCallFrame>,
    /// Root frames (completed top-level frames).
    root_frames: Vec<TracedCallFrame>,
    /// Previous register state for diff computation.
    prev_registers: [u64; 12],
    /// Program ID for trace context.
    program_id: Option<String>,
    /// Optional CFG builder for control flow analysis.
    cfg_builder: Option<CfgBuilder>,
    /// Optional data flow analyzer.
    dataflow_analyzer: Option<DataFlowAnalyzer>,
    /// Execution sequence counter for instruction ordering.
    /// Monotonically increasing within a program invocation, resets on CPI.
    exec_seq_counter: u64,
}

impl Tracer {
    /// Create a new tracer with optional symbol map and program ID.
    pub fn new(symbol_map: HashMap<u64, String>, program_id: Option<String>) -> Self {
        Self {
            symbol_map,
            decoder_registry: SyscallDecoderRegistry::new(),
            frame_stack: Vec::new(),
            root_frames: Vec::new(),
            prev_registers: [0u64; 12],
            program_id,
            cfg_builder: None,
            dataflow_analyzer: None,
            exec_seq_counter: 0,
        }
    }

    /// Enable CFG analysis during tracing.
    pub fn enable_cfg_analysis(&mut self) {
        self.cfg_builder = Some(CfgBuilder::new(self.symbol_map.clone()));
    }

    /// Disable CFG analysis.
    pub fn disable_cfg_analysis(&mut self) {
        self.cfg_builder = None;
    }

    /// Check if CFG analysis is enabled.
    pub fn cfg_enabled(&self) -> bool {
        self.cfg_builder.is_some()
    }

    /// Enable data flow analysis during tracing.
    ///
    /// If `track_all` is true, all definitions are tracked.
    /// Otherwise, only tainted values are tracked.
    pub fn enable_dataflow(&mut self, track_all: bool) {
        self.dataflow_analyzer = Some(DataFlowAnalyzer::new(track_all));
    }

    /// Disable data flow analysis.
    pub fn disable_dataflow(&mut self) {
        self.dataflow_analyzer = None;
    }

    /// Check if data flow analysis is enabled.
    pub fn dataflow_enabled(&self) -> bool {
        self.dataflow_analyzer.is_some()
    }

    /// Build a tracer from an executable's function registry.
    ///
    /// This extracts symbol names from the executable for better trace output.
    pub fn from_executable<C: ContextObject>(
        executable: &Executable<C>,
        program_id: Option<String>,
    ) -> Self {
        let mut symbol_map = HashMap::new();

        // Extract function names from the executable's function registry
        for (_key, (name, pc)) in executable.get_function_registry().iter() {
            let name_str = String::from_utf8_lossy(name).to_string();
            // Only add non-empty names
            if !name_str.is_empty() {
                symbol_map.insert(pc as u64, name_str);
            }
        }

        // Also extract syscall names from the loader's registry
        // These are stored by hash key, not PC, so we can't map them to PCs directly,
        // but we can use them for syscall decoding
        for (key, (name, _addr)) in executable.get_loader().get_function_registry().iter() {
            let name_str = String::from_utf8_lossy(name).to_string();
            if !name_str.is_empty() {
                // Store with key as a pseudo-PC for syscall lookup
                // (syscall instructions use the key as immediate value)
                symbol_map.insert(key as u64 | 0x8000_0000_0000_0000, name_str);
            }
        }

        Self::new(symbol_map, program_id)
    }

    /// Register a custom syscall decoder.
    pub fn register_syscall_decoder(&mut self, decoder: Box<dyn SyscallDecoder>) {
        self.decoder_registry.register(decoder);
    }

    /// Add a symbol mapping (PC -> function name).
    pub fn add_symbol(&mut self, pc: u64, name: String) {
        self.symbol_map.insert(pc, name);
    }

    /// Execute the program with tracing enabled.
    ///
    /// Returns the instruction count, program result, and trace context.
    /// This only works in interpreter mode (not JIT).
    pub fn execute<'a, 'b, C: ContextObject>(
        &mut self,
        vm: &'a mut EbpfVm<'b, C>,
        executable: &'a Executable<C>,
    ) -> (u64, ProgramResult, TraceContext) {
        // Reset tracer state
        self.frame_stack.clear();
        self.root_frames.clear();
        self.exec_seq_counter = 0;

        // Initialize VM state (similar to execute_program)
        let initial_pc = executable.get_entrypoint_instruction_offset() as u64;
        vm.registers[11] = initial_pc;
        let config = executable.get_config();
        let initial_insn_count = vm.context_object_pointer.get_remaining();
        vm.previous_instruction_meter = initial_insn_count;
        vm.due_insn_count = 0;
        vm.program_result = ProgramResult::Ok(0);

        self.prev_registers = vm.registers;

        // Reinitialize CFG builder if enabled
        if self.cfg_builder.is_some() {
            self.cfg_builder = Some(CfgBuilder::new(self.symbol_map.clone()));
        }

        // Initialize dataflow analyzer if enabled
        if let Some(ref mut analyzer) = self.dataflow_analyzer {
            analyzer.init(&vm.registers);
        }

        // Push initial frame
        let entry_symbol = self
            .symbol_map
            .get(&initial_pc)
            .cloned()
            .or_else(|| Some("entrypoint".to_string()));
        self.frame_stack.push(TracedCallFrame::new(entry_symbol, initial_pc, 0));

        // Create interpreter
        let mut interpreter = Interpreter::new(vm, executable, vm.registers);

        // Execute with tracing
        while self.traced_step(&mut interpreter, executable) {}

        // Finalize - consume remaining instruction count
        let instruction_count = if config.enable_instruction_meter {
            interpreter
                .vm
                .context_object_pointer
                .consume(interpreter.vm.due_insn_count);
            initial_insn_count.saturating_sub(
                interpreter
                    .vm
                    .context_object_pointer
                    .get_remaining(),
            )
        } else {
            0
        };

        // Build trace result
        let result = match &interpreter.vm.program_result {
            ProgramResult::Ok(val) => TraceResult::Success { return_value: *val },
            ProgramResult::Err(e) => TraceResult::Error {
                message: format!("{:?}", e),
            },
        };

        // Pop remaining frames to root
        while let Some(frame) = self.frame_stack.pop() {
            if let Some(parent) = self.frame_stack.last_mut() {
                parent.sub_calls.push(frame);
            } else {
                self.root_frames.push(frame);
            }
        }

        // Build CFG if enabled
        let control_flow_graph = self.cfg_builder.take().map(|builder| builder.build());

        // Finalize dataflow if enabled
        let dataflow = self.dataflow_analyzer.take().map(|analyzer| analyzer.finalize());

        // Get text section vaddr for address calculation
        let (text_section_vaddr, _) = executable.get_text_bytes();

        let trace_context = TraceContext {
            program_id: self.program_id.clone(),
            execution_tree: std::mem::take(&mut self.root_frames),
            total_compute_units: instruction_count,
            result,
            text_section_vaddr,
            control_flow_graph,
            dataflow,
        };

        // Extract program result
        let mut prog_result = ProgramResult::Ok(0);
        std::mem::swap(&mut prog_result, &mut interpreter.vm.program_result);

        (instruction_count, prog_result, trace_context)
    }

    /// Execute one instruction with tracing.
    fn traced_step<'a, 'b, C: ContextObject>(
        &mut self,
        interpreter: &mut Interpreter<'a, 'b, C>,
        executable: &Executable<C>,
    ) -> bool {
        let pc = interpreter.reg[11];
        let (_program_vm_addr, program) = executable.get_text_bytes();

        // Bounds check
        if (pc as usize) * ebpf::INSN_SIZE >= program.len() {
            return false;
        }

        // Get instruction before execution
        let insn = ebpf::get_insn_unchecked(program, pc as usize);
        let opcode = insn.opc;

        // Capture pre-execution state
        let pre_regs = interpreter.reg;
        let pre_cu = interpreter.vm.context_object_pointer.get_remaining();
        let pre_call_depth = interpreter.vm.call_depth;

        // Detect instruction type before execution
        // Syscall detection matches disassembler logic:
        // - insn.src == 0 means it's definitely a syscall
        // - OR if function is not found in program registry AND not using static syscalls
        let is_syscall = if opcode == ebpf::CALL_IMM {
            if insn.src == 0 {
                true
            } else {
                // Check if this is a call to a function or a syscall
                let key = executable.get_sbpf_version().calculate_call_imm_target_pc(pc as usize, insn.imm);
                let function_found = executable.get_function_registry().lookup_by_key(key).is_some();
                !function_found && !executable.get_sbpf_version().static_syscalls()
            }
        } else {
            false
        };
        let is_internal_call = opcode == ebpf::CALL_IMM && !is_syscall;
        let is_call_reg = opcode == ebpf::CALL_REG;
        let is_exit = opcode == ebpf::EXIT;

        // Execute the instruction
        let continue_execution = interpreter.step();

        // Capture post-execution state
        let post_regs = interpreter.reg;
        let post_cu = interpreter.vm.context_object_pointer.get_remaining();
        let post_call_depth = interpreter.vm.call_depth;
        let cu_consumed = pre_cu.saturating_sub(post_cu);

        // Generate mnemonic
        let mnemonic = disassemble_instruction(
            &insn,
            pc as usize,
            &BTreeMap::new(),
            executable.get_function_registry(),
            executable.get_loader(),
            executable.get_sbpf_version(),
        );

        // Compute register changes with before/after values
        let register_changes = self.compute_reg_diff(&pre_regs, &post_regs);

        // Increment execution sequence counter
        self.exec_seq_counter += 1;
        let exec_seq = self.exec_seq_counter;

        // Extract operand information for dataflow tracing
        let (operand_1, operand_2, immediate) = self.extract_operands(&insn, opcode, &pre_regs);

        // Record instruction event
        let insn_event = InstructionEvent {
            exec_seq,
            pc,
            opcode,
            mnemonic,
            register_changes,
            compute_units: cu_consumed,
            operand_1,
            operand_2,
            immediate,
        };

        if let Some(current_frame) = self.frame_stack.last_mut() {
            current_frame
                .events
                .push(TraceEvent::Instruction(insn_event));
            current_frame.frame_compute_units = current_frame
                .frame_compute_units
                .saturating_add(cu_consumed);
        }

        // Record memory access for load/store instructions
        let mem_info = self.record_memory_access(&insn, &pre_regs, &interpreter.vm.memory_mapping);

        // Feed CFG builder
        if let Some(ref mut cfg_builder) = self.cfg_builder {
            cfg_builder.record_instruction(pc, opcode, &insn, &pre_regs, &post_regs);
        }

        // Feed dataflow analyzer
        if let Some(ref mut analyzer) = self.dataflow_analyzer {
            let (mem_region, mem_addr, mem_size) = mem_info
                .map(|(r, a, s)| (Some(r), Some(a), Some(s)))
                .unwrap_or((None, None, None));
            analyzer.analyze_instruction(
                exec_seq,
                pc,
                opcode,
                &insn,
                &pre_regs,
                &post_regs,
                mem_region,
                mem_addr,
                mem_size,
            );
        }

        // Handle syscall
        if is_syscall {
            self.record_syscall(
                &insn,
                &pre_regs,
                &post_regs,
                &interpreter.vm.memory_mapping,
                executable,
                cu_consumed,
            );

            // Record syscall in dataflow
            if let Some(ref mut analyzer) = self.dataflow_analyzer {
                let syscall_key = insn.imm as u32;
                let syscall_name = executable
                    .get_loader()
                    .get_function_registry()
                    .lookup_by_key(syscall_key)
                    .map(|(name, _)| String::from_utf8_lossy(name).to_string())
                    .unwrap_or_else(|| format!("syscall_{:#x}", syscall_key));
                analyzer.handle_syscall(pc, &syscall_name, &post_regs);
            }
        }

        // Handle call frame management
        if (is_internal_call || is_call_reg) && post_call_depth > pre_call_depth {
            // New frame was pushed - record FunctionCall event
            let target_pc = post_regs[11];
            let symbol = self.symbol_map.get(&target_pc).cloned();
            let new_depth = self.frame_stack.len() as u32;

            // Record the function call event in the current (caller) frame
            let call_event = FunctionCallEvent {
                target_function: symbol.clone(),
                target_pc,
                call_site_pc: pc,
                is_indirect: is_call_reg,
                args: [pre_regs[1], pre_regs[2], pre_regs[3], pre_regs[4], pre_regs[5]],
                depth: new_depth,
            };
            if let Some(current_frame) = self.frame_stack.last_mut() {
                current_frame.events.push(TraceEvent::FunctionCall(call_event));
            }

            // Push new frame for the callee
            self.frame_stack
                .push(TracedCallFrame::new(symbol, target_pc, new_depth));
        } else if is_exit && post_call_depth < pre_call_depth {
            // Frame was popped (return from function)
            if let Some(frame) = self.frame_stack.pop() {
                // Record function return event
                let return_event = FunctionReturnEvent {
                    from_function: frame.symbol_name.clone(),
                    return_value: post_regs[0],
                    return_pc: post_regs[11],
                    depth: self.frame_stack.len() as u32,
                };

                if let Some(parent) = self.frame_stack.last_mut() {
                    parent.events.push(TraceEvent::FunctionReturn(return_event));
                    parent.sub_calls.push(frame);
                } else {
                    self.root_frames.push(frame);
                }
            }
        }

        self.prev_registers = post_regs;
        continue_execution
    }

    /// Compute register differences with before/after values.
    fn compute_reg_diff(&self, pre: &[u64; 12], post: &[u64; 12]) -> Vec<RegisterChange> {
        let mut changes = Vec::new();
        for i in 0..12 {
            if pre[i] != post[i] {
                changes.push(RegisterChange {
                    register: i as u8,
                    value_before: pre[i],
                    value_after: post[i],
                });
            }
        }
        changes
    }

    /// Extract operand information for an instruction.
    /// Returns (operand_1, operand_2, immediate) where:
    /// - operand_1 is typically the dst register's value before execution
    /// - operand_2 is the src register's value (for reg-reg ops) or None
    /// - immediate is the immediate value if the instruction uses one
    fn extract_operands(
        &self,
        insn: &Insn,
        opcode: u8,
        pre_regs: &[u64; 12],
    ) -> (Option<OperandInfo>, Option<OperandInfo>, Option<i64>) {
        // Get DefIds for registers if dataflow is enabled
        let dst_def = self.dataflow_analyzer.as_ref()
            .and_then(|a| a.state.get_reg_def(insn.dst));
        let src_def = self.dataflow_analyzer.as_ref()
            .and_then(|a| a.state.get_reg_def(insn.src));

        // Classify instruction type by opcode
        match opcode {
            // ALU operations with immediate: dst op= imm
            ebpf::ADD32_IMM | ebpf::ADD64_IMM |
            ebpf::SUB32_IMM | ebpf::SUB64_IMM |
            ebpf::MUL32_IMM | ebpf::MUL64_IMM |
            ebpf::DIV32_IMM | ebpf::DIV64_IMM |
            ebpf::MOD32_IMM | ebpf::MOD64_IMM |
            ebpf::OR32_IMM | ebpf::OR64_IMM |
            ebpf::AND32_IMM | ebpf::AND64_IMM |
            ebpf::XOR32_IMM | ebpf::XOR64_IMM |
            ebpf::LSH32_IMM | ebpf::LSH64_IMM |
            ebpf::RSH32_IMM | ebpf::RSH64_IMM |
            ebpf::ARSH32_IMM | ebpf::ARSH64_IMM => {
                let op1 = Some(OperandInfo {
                    value: pre_regs[insn.dst as usize],
                    source_reg: Some(insn.dst),
                    source_def: dst_def,
                    is_immediate: false,
                });
                let op2 = Some(OperandInfo {
                    value: insn.imm as u64,
                    source_reg: None,
                    source_def: None,
                    is_immediate: true,
                });
                (op1, op2, Some(insn.imm))
            }

            // ALU operations with register: dst op= src
            ebpf::ADD32_REG | ebpf::ADD64_REG |
            ebpf::SUB32_REG | ebpf::SUB64_REG |
            ebpf::MUL32_REG | ebpf::MUL64_REG |
            ebpf::DIV32_REG | ebpf::DIV64_REG |
            ebpf::MOD32_REG | ebpf::MOD64_REG |
            ebpf::OR32_REG | ebpf::OR64_REG |
            ebpf::AND32_REG | ebpf::AND64_REG |
            ebpf::XOR32_REG | ebpf::XOR64_REG |
            ebpf::LSH32_REG | ebpf::LSH64_REG |
            ebpf::RSH32_REG | ebpf::RSH64_REG |
            ebpf::ARSH32_REG | ebpf::ARSH64_REG => {
                let op1 = Some(OperandInfo {
                    value: pre_regs[insn.dst as usize],
                    source_reg: Some(insn.dst),
                    source_def: dst_def,
                    is_immediate: false,
                });
                let op2 = Some(OperandInfo {
                    value: pre_regs[insn.src as usize],
                    source_reg: Some(insn.src),
                    source_def: src_def,
                    is_immediate: false,
                });
                (op1, op2, None)
            }

            // MOV immediate
            ebpf::MOV32_IMM | ebpf::MOV64_IMM => {
                let op2 = Some(OperandInfo {
                    value: insn.imm as u64,
                    source_reg: None,
                    source_def: None,
                    is_immediate: true,
                });
                (None, op2, Some(insn.imm))
            }

            // MOV register
            ebpf::MOV32_REG | ebpf::MOV64_REG => {
                let op2 = Some(OperandInfo {
                    value: pre_regs[insn.src as usize],
                    source_reg: Some(insn.src),
                    source_def: src_def,
                    is_immediate: false,
                });
                (None, op2, None)
            }

            // NEG (unary)
            ebpf::NEG32 | ebpf::NEG64 => {
                let op1 = Some(OperandInfo {
                    value: pre_regs[insn.dst as usize],
                    source_reg: Some(insn.dst),
                    source_def: dst_def,
                    is_immediate: false,
                });
                (op1, None, None)
            }

            // Load double-word immediate
            ebpf::LD_DW_IMM => {
                // The immediate is a 64-bit value split across two instructions
                // We just capture what we can here
                (None, None, Some(insn.imm))
            }

            // Memory loads: dst = [src + off]
            ebpf::LD_B_REG | ebpf::LD_H_REG | ebpf::LD_W_REG | ebpf::LD_DW_REG |
            ebpf::LD_1B_REG | ebpf::LD_2B_REG | ebpf::LD_4B_REG | ebpf::LD_8B_REG => {
                // src register holds base address
                let op1 = Some(OperandInfo {
                    value: pre_regs[insn.src as usize],
                    source_reg: Some(insn.src),
                    source_def: src_def,
                    is_immediate: false,
                });
                // offset is an immediate
                (op1, None, Some(insn.off as i64))
            }

            // Memory stores: [dst + off] = src or imm
            ebpf::ST_B_REG | ebpf::ST_H_REG | ebpf::ST_W_REG | ebpf::ST_DW_REG |
            ebpf::ST_1B_REG | ebpf::ST_2B_REG | ebpf::ST_4B_REG | ebpf::ST_8B_REG => {
                // dst register holds base address, src holds value
                let op1 = Some(OperandInfo {
                    value: pre_regs[insn.dst as usize],
                    source_reg: Some(insn.dst),
                    source_def: dst_def,
                    is_immediate: false,
                });
                let op2 = Some(OperandInfo {
                    value: pre_regs[insn.src as usize],
                    source_reg: Some(insn.src),
                    source_def: src_def,
                    is_immediate: false,
                });
                (op1, op2, Some(insn.off as i64))
            }

            ebpf::ST_B_IMM | ebpf::ST_H_IMM | ebpf::ST_W_IMM | ebpf::ST_DW_IMM |
            ebpf::ST_1B_IMM | ebpf::ST_2B_IMM | ebpf::ST_4B_IMM | ebpf::ST_8B_IMM => {
                // dst register holds base address, imm is the value
                let op1 = Some(OperandInfo {
                    value: pre_regs[insn.dst as usize],
                    source_reg: Some(insn.dst),
                    source_def: dst_def,
                    is_immediate: false,
                });
                let op2 = Some(OperandInfo {
                    value: insn.imm as u64,
                    source_reg: None,
                    source_def: None,
                    is_immediate: true,
                });
                (op1, op2, Some(insn.imm))
            }

            // For other instructions (jumps, calls, etc.), no operand extraction
            _ => (None, None, None),
        }
    }

    /// Record memory access event for load/store instructions.
    /// Returns (region, address, size) if this was a memory instruction.
    fn record_memory_access(
        &mut self,
        insn: &Insn,
        regs: &[u64; 12],
        memory_mapping: &MemoryMapping,
    ) -> Option<(MemoryRegionType, u64, u64)> {
        let (action, addr, size) = match insn.opc {
            // Load instructions (old style)
            ebpf::LD_B_REG => {
                let addr = (regs[insn.src as usize] as i64).wrapping_add(insn.off as i64) as u64;
                (MemoryAccessType::Load, addr, 1u64)
            }
            ebpf::LD_H_REG => {
                let addr = (regs[insn.src as usize] as i64).wrapping_add(insn.off as i64) as u64;
                (MemoryAccessType::Load, addr, 2u64)
            }
            ebpf::LD_W_REG => {
                let addr = (regs[insn.src as usize] as i64).wrapping_add(insn.off as i64) as u64;
                (MemoryAccessType::Load, addr, 4u64)
            }
            ebpf::LD_DW_REG => {
                let addr = (regs[insn.src as usize] as i64).wrapping_add(insn.off as i64) as u64;
                (MemoryAccessType::Load, addr, 8u64)
            }
            // Load instructions (SBPFv3 style)
            ebpf::LD_1B_REG => {
                let addr = (regs[insn.src as usize] as i64).wrapping_add(insn.off as i64) as u64;
                (MemoryAccessType::Load, addr, 1u64)
            }
            ebpf::LD_2B_REG => {
                let addr = (regs[insn.src as usize] as i64).wrapping_add(insn.off as i64) as u64;
                (MemoryAccessType::Load, addr, 2u64)
            }
            ebpf::LD_4B_REG => {
                let addr = (regs[insn.src as usize] as i64).wrapping_add(insn.off as i64) as u64;
                (MemoryAccessType::Load, addr, 4u64)
            }
            ebpf::LD_8B_REG => {
                let addr = (regs[insn.src as usize] as i64).wrapping_add(insn.off as i64) as u64;
                (MemoryAccessType::Load, addr, 8u64)
            }
            // Store instructions (immediate, old style)
            ebpf::ST_B_IMM | ebpf::ST_H_IMM | ebpf::ST_W_IMM | ebpf::ST_DW_IMM => {
                let addr = (regs[insn.dst as usize] as i64).wrapping_add(insn.off as i64) as u64;
                let size = match insn.opc {
                    ebpf::ST_B_IMM => 1u64,
                    ebpf::ST_H_IMM => 2u64,
                    ebpf::ST_W_IMM => 4u64,
                    ebpf::ST_DW_IMM => 8u64,
                    _ => return None,
                };
                (MemoryAccessType::Store, addr, size)
            }
            // Store instructions (register, old style)
            ebpf::ST_B_REG | ebpf::ST_H_REG | ebpf::ST_W_REG | ebpf::ST_DW_REG => {
                let addr = (regs[insn.dst as usize] as i64).wrapping_add(insn.off as i64) as u64;
                let size = match insn.opc {
                    ebpf::ST_B_REG => 1u64,
                    ebpf::ST_H_REG => 2u64,
                    ebpf::ST_W_REG => 4u64,
                    ebpf::ST_DW_REG => 8u64,
                    _ => return None,
                };
                (MemoryAccessType::Store, addr, size)
            }
            // Store instructions (SBPFv3 immediate)
            ebpf::ST_1B_IMM | ebpf::ST_2B_IMM | ebpf::ST_4B_IMM | ebpf::ST_8B_IMM => {
                let addr = (regs[insn.dst as usize] as i64).wrapping_add(insn.off as i64) as u64;
                let size = match insn.opc {
                    ebpf::ST_1B_IMM => 1u64,
                    ebpf::ST_2B_IMM => 2u64,
                    ebpf::ST_4B_IMM => 4u64,
                    ebpf::ST_8B_IMM => 8u64,
                    _ => return None,
                };
                (MemoryAccessType::Store, addr, size)
            }
            // Store instructions (SBPFv3 register)
            ebpf::ST_1B_REG | ebpf::ST_2B_REG | ebpf::ST_4B_REG | ebpf::ST_8B_REG => {
                let addr = (regs[insn.dst as usize] as i64).wrapping_add(insn.off as i64) as u64;
                let size = match insn.opc {
                    ebpf::ST_1B_REG => 1u64,
                    ebpf::ST_2B_REG => 2u64,
                    ebpf::ST_4B_REG => 4u64,
                    ebpf::ST_8B_REG => 8u64,
                    _ => return None,
                };
                (MemoryAccessType::Store, addr, size)
            }
            _ => return None, // Not a memory instruction
        };

        let region = classify_pointer(addr);

        // Try to read the value for hex encoding
        let read_len = size.min(MAX_MEMORY_VALUE_BYTES as u64);
        let (value_hex, truncated) = match memory_mapping
            .map(AccessType::Load, addr, read_len)
            .into()
        {
            Ok(host_addr) => {
                let bytes =
                    unsafe { std::slice::from_raw_parts(host_addr as *const u8, read_len as usize) };
                format_hex_value(bytes, MAX_MEMORY_VALUE_BYTES)
            }
            Err(_) => ("(inaccessible)".to_string(), false),
        };

        let event = MemoryAccessEvent {
            region,
            action,
            address: addr,
            size,
            value_hex,
            truncated,
        };

        if let Some(current_frame) = self.frame_stack.last_mut() {
            current_frame.events.push(TraceEvent::MemoryAccess(event));
        }

        Some((region, addr, size))
    }

    /// Record syscall event.
    fn record_syscall<C: ContextObject>(
        &mut self,
        insn: &Insn,
        pre_regs: &[u64; 12],
        post_regs: &[u64; 12],
        memory_mapping: &MemoryMapping,
        executable: &Executable<C>,
        cu_consumed: u64,
    ) {
        // Look up syscall name from loader's function registry
        let syscall_key = insn.imm as u32;
        let syscall_name = executable
            .get_loader()
            .get_function_registry()
            .lookup_by_key(syscall_key)
            .map(|(name, _)| String::from_utf8_lossy(name).to_string())
            .unwrap_or_else(|| format!("syscall_{:#x}", syscall_key));

        let raw_args = [pre_regs[1], pre_regs[2], pre_regs[3], pre_regs[4], pre_regs[5]];
        let args_decoded = self
            .decoder_registry
            .decode(&syscall_name, raw_args, memory_mapping);

        let event = SyscallEvent {
            name: syscall_name,
            args_decoded,
            raw_args,
            return_value: Some(post_regs[0]),
            compute_units: cu_consumed,
        };

        if let Some(current_frame) = self.frame_stack.last_mut() {
            current_frame.events.push(TraceEvent::Syscall(event));
        }
    }

    /// Export trace context as JSON string.
    pub fn to_json(context: &TraceContext) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(context)
    }

    /// Export trace context as compact JSON string.
    pub fn to_json_compact(context: &TraceContext) -> Result<String, serde_json::Error> {
        serde_json::to_string(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracer_new() {
        let tracer = Tracer::new(HashMap::new(), Some("test_program".to_string()));
        assert_eq!(tracer.program_id, Some("test_program".to_string()));
        assert!(tracer.symbol_map.is_empty());
    }

    #[test]
    fn test_add_symbol() {
        let mut tracer = Tracer::new(HashMap::new(), None);
        tracer.add_symbol(100, "my_function".to_string());
        assert_eq!(tracer.symbol_map.get(&100), Some(&"my_function".to_string()));
    }

    #[test]
    fn test_compute_reg_diff() {
        let tracer = Tracer::new(std::collections::HashMap::new(), None);
        let pre = [0u64; 12];
        let mut post = [0u64; 12];
        post[0] = 42; // r0 changed
        post[5] = 100; // r5 changed

        let changes = tracer.compute_reg_diff(&pre, &post);
        assert!(changes.iter().any(|rc| rc.register == 0 && rc.value_before == 0 && rc.value_after == 42));
        assert!(changes.iter().any(|rc| rc.register == 5 && rc.value_before == 0 && rc.value_after == 100));
        assert!(!changes.iter().any(|rc| rc.register == 1)); // unchanged
    }
}
