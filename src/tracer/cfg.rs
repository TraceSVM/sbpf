//! Control Flow Graph (CFG) analysis for dynamic execution traces.
//!
//! This module provides runtime CFG collection during traced execution:
//! - Basic block detection with terminator classification
//! - Edge tracking (unconditional, conditional, back edges)
//! - Loop detection with iteration counting
//! - Branch condition inference

use crate::ebpf::{self, Insn};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// A basic block identified during execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicBlock {
    /// Starting PC of the basic block (inclusive).
    pub start_pc: u64,
    /// Ending PC of the basic block (inclusive) - the terminator instruction.
    pub end_pc: u64,
    /// Number of instructions in this block.
    pub instruction_count: u32,
    /// Type of terminator that ends this block.
    pub terminator: BlockTerminator,
    /// Human-readable label (from symbol map).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Number of times this block was executed.
    pub execution_count: u32,
}

/// Type of instruction that terminates a basic block.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum BlockTerminator {
    /// Unconditional jump (ja instruction).
    UnconditionalJump,
    /// Conditional branch (jeq, jne, jgt, etc.).
    ConditionalBranch {
        /// The comparison operation.
        condition: BranchCondition,
        /// Whether it compares against immediate or register.
        is_immediate: bool,
        /// Whether it's a 32-bit comparison.
        is_32bit: bool,
    },
    /// Function call (call imm or callx).
    FunctionCall {
        /// Whether this is an indirect call (callx).
        is_indirect: bool,
    },
    /// Function return (exit instruction).
    Exit,
    /// Block ends because next instruction is a jump target (fall-through).
    FallThrough,
}

/// Branch condition type for conditional jumps.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BranchCondition {
    /// Equal (jeq)
    Equal,
    /// Not equal (jne)
    NotEqual,
    /// Greater than unsigned (jgt)
    GreaterThan,
    /// Greater or equal unsigned (jge)
    GreaterOrEqual,
    /// Less than unsigned (jlt)
    LessThan,
    /// Less or equal unsigned (jle)
    LessOrEqual,
    /// Greater than signed (jsgt)
    SignedGreaterThan,
    /// Greater or equal signed (jsge)
    SignedGreaterOrEqual,
    /// Less than signed (jslt)
    SignedLessThan,
    /// Less or equal signed (jsle)
    SignedLessOrEqual,
    /// Bitwise AND test (jset)
    BitTest,
}

/// An edge in the control flow graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfgEdge {
    /// Source basic block (by start_pc).
    pub from_block: u64,
    /// Target basic block (by start_pc).
    pub to_block: u64,
    /// Type of edge.
    pub edge_type: EdgeType,
    /// Number of times this edge was taken during execution.
    pub execution_count: u32,
    /// Branch condition info if conditional.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch_info: Option<BranchTakenInfo>,
}

/// Type of CFG edge.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EdgeType {
    /// Unconditional jump or fall-through.
    Unconditional,
    /// Conditional branch taken (condition was true).
    ConditionalTaken,
    /// Conditional branch not taken (fall-through).
    ConditionalFallthrough,
    /// Function call edge (to callee).
    Call,
    /// Function return edge (back to caller).
    Return,
    /// Back edge (target PC <= source PC) - indicates a loop.
    BackEdge,
}

/// Information about a conditional branch that was taken/not taken.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchTakenInfo {
    /// The condition being tested.
    pub condition: BranchCondition,
    /// Register being compared (destination register).
    pub dst_register: u8,
    /// Value in the destination register at branch time.
    pub dst_value: u64,
    /// Source register or immediate value.
    pub comparand: BranchComparand,
    /// Whether the branch was taken.
    pub was_taken: bool,
}

/// The value being compared against in a branch.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum BranchComparand {
    /// Immediate value.
    Immediate { value: i64 },
    /// Register value.
    Register { reg: u8, value: u64 },
}

/// A detected loop in the control flow graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedLoop {
    /// PC of the loop header (target of back edge).
    pub header_pc: u64,
    /// PC of the back edge source (the instruction jumping back).
    pub back_edge_pc: u64,
    /// All basic blocks that are part of this loop body.
    pub body_blocks: Vec<u64>,
    /// Number of iterations observed during execution.
    pub iteration_count: u32,
    /// Estimated loop bound if detectable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_bound: Option<LoopBound>,
}

/// Estimated loop bound information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopBound {
    /// The register that appears to be the loop counter.
    pub counter_register: u8,
    /// Initial value of the counter.
    pub initial_value: u64,
    /// Final value when loop exits.
    pub final_value: u64,
    /// Step size per iteration (positive or negative).
    pub step: i64,
}

/// Complete CFG analysis result for a traced execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ControlFlowGraph {
    /// Basic blocks discovered during execution, keyed by start PC.
    pub basic_blocks: BTreeMap<u64, BasicBlock>,
    /// Edges between basic blocks.
    pub edges: Vec<CfgEdge>,
    /// Detected loops (by header PC).
    pub loops: Vec<DetectedLoop>,
    /// Entry point PC.
    pub entry_pc: u64,
    /// Exit PCs (blocks that end with exit instruction).
    pub exit_pcs: Vec<u64>,
}

/// Builder that collects CFG information during execution.
pub struct CfgBuilder {
    /// Current basic block start PC.
    current_block_start: Option<u64>,
    /// Instructions in the current block.
    current_block_pcs: Vec<u64>,
    /// Completed basic blocks.
    blocks: BTreeMap<u64, BasicBlock>,
    /// Set of all PCs that are jump targets (for block boundary detection).
    jump_targets: BTreeSet<u64>,
    /// Edge counts for deduplication: (from_block, to_block) -> count.
    edge_counts: HashMap<(u64, u64), (u32, EdgeType, Option<BranchTakenInfo>)>,
    /// Back edges detected (source_pc -> target_pc).
    back_edges: Vec<(u64, u64)>,
    /// Loop iteration counters (header_pc -> count).
    loop_iterations: HashMap<u64, u32>,
    /// Symbol map for labeling.
    symbol_map: HashMap<u64, String>,
    /// Entry PC.
    entry_pc: u64,
    /// Last terminator PC for edge tracking.
    last_terminator_pc: Option<u64>,
    /// Last terminator type.
    last_terminator_type: Option<BlockTerminator>,
    /// Branch history for loop bound inference.
    branch_history: Vec<BranchTakenInfo>,
}

impl CfgBuilder {
    /// Create a new CFG builder.
    pub fn new(symbol_map: HashMap<u64, String>) -> Self {
        Self {
            current_block_start: None,
            current_block_pcs: Vec::new(),
            blocks: BTreeMap::new(),
            jump_targets: BTreeSet::new(),
            edge_counts: HashMap::new(),
            back_edges: Vec::new(),
            loop_iterations: HashMap::new(),
            symbol_map,
            entry_pc: 0,
            last_terminator_pc: None,
            last_terminator_type: None,
            branch_history: Vec::new(),
        }
    }

    /// Record an instruction execution.
    pub fn record_instruction(
        &mut self,
        pc: u64,
        opcode: u8,
        insn: &Insn,
        pre_regs: &[u64; 12],
        post_regs: &[u64; 12],
    ) {
        // Set entry PC on first instruction
        if self.current_block_start.is_none() && self.blocks.is_empty() {
            self.entry_pc = pc;
        }

        // Check if we should start a new block
        let should_start_new_block = self.current_block_start.is_none()
            || self.jump_targets.contains(&pc);

        if should_start_new_block {
            // Finalize previous block if any
            if self.current_block_start.is_some() {
                self.finalize_current_block(BlockTerminator::FallThrough);
            }

            // Start new block
            self.current_block_start = Some(pc);
            self.current_block_pcs.clear();
        }

        // Add instruction to current block
        self.current_block_pcs.push(pc);

        // Check if this instruction is a block terminator
        if Self::is_block_terminator(opcode) {
            let terminator = self.classify_terminator(opcode, insn);

            // Record branch condition if conditional
            if Self::is_conditional_branch(opcode) {
                self.record_branch_info(opcode, insn, pre_regs, post_regs);
            }

            // Mark jump target for future block detection
            if opcode != ebpf::EXIT {
                let next_pc = pc + 1;
                let target = self.get_branch_target(pc, insn, opcode);

                // Add target as jump target
                if let Some(target_pc) = target {
                    self.jump_targets.insert(target_pc);

                    // Detect back edge (loop)
                    if target_pc <= pc {
                        self.back_edges.push((pc, target_pc));
                        *self.loop_iterations.entry(target_pc).or_insert(0) += 1;
                    }
                }

                // Fall-through is also a jump target for conditional branches
                if Self::is_conditional_branch(opcode) {
                    self.jump_targets.insert(next_pc);
                }
            }

            self.finalize_current_block(terminator.clone());

            // Record edge based on where we actually went
            let actual_target = post_regs[11];
            self.record_edge(pc, actual_target, opcode, insn, pre_regs, post_regs);

            self.last_terminator_pc = Some(pc);
            self.last_terminator_type = Some(terminator);
        }
    }

    /// Check if opcode is a block terminator.
    fn is_block_terminator(opcode: u8) -> bool {
        matches!(
            opcode,
            ebpf::JA
                | ebpf::JEQ64_IMM
                | ebpf::JEQ64_REG
                | ebpf::JNE64_IMM
                | ebpf::JNE64_REG
                | ebpf::JGT64_IMM
                | ebpf::JGT64_REG
                | ebpf::JGE64_IMM
                | ebpf::JGE64_REG
                | ebpf::JLT64_IMM
                | ebpf::JLT64_REG
                | ebpf::JLE64_IMM
                | ebpf::JLE64_REG
                | ebpf::JSGT64_IMM
                | ebpf::JSGT64_REG
                | ebpf::JSGE64_IMM
                | ebpf::JSGE64_REG
                | ebpf::JSLT64_IMM
                | ebpf::JSLT64_REG
                | ebpf::JSLE64_IMM
                | ebpf::JSLE64_REG
                | ebpf::JSET64_IMM
                | ebpf::JSET64_REG
                | ebpf::JEQ32_IMM
                | ebpf::JEQ32_REG
                | ebpf::JNE32_IMM
                | ebpf::JNE32_REG
                | ebpf::JGT32_IMM
                | ebpf::JGT32_REG
                | ebpf::JGE32_IMM
                | ebpf::JGE32_REG
                | ebpf::JLT32_IMM
                | ebpf::JLT32_REG
                | ebpf::JLE32_IMM
                | ebpf::JLE32_REG
                | ebpf::JSGT32_IMM
                | ebpf::JSGT32_REG
                | ebpf::JSGE32_IMM
                | ebpf::JSGE32_REG
                | ebpf::JSLT32_IMM
                | ebpf::JSLT32_REG
                | ebpf::JSLE32_IMM
                | ebpf::JSLE32_REG
                | ebpf::JSET32_IMM
                | ebpf::JSET32_REG
                | ebpf::CALL_IMM
                | ebpf::CALL_REG
                | ebpf::EXIT
        )
    }

    /// Check if opcode is a conditional branch.
    fn is_conditional_branch(opcode: u8) -> bool {
        matches!(
            opcode,
            ebpf::JEQ64_IMM
                | ebpf::JEQ64_REG
                | ebpf::JNE64_IMM
                | ebpf::JNE64_REG
                | ebpf::JGT64_IMM
                | ebpf::JGT64_REG
                | ebpf::JGE64_IMM
                | ebpf::JGE64_REG
                | ebpf::JLT64_IMM
                | ebpf::JLT64_REG
                | ebpf::JLE64_IMM
                | ebpf::JLE64_REG
                | ebpf::JSGT64_IMM
                | ebpf::JSGT64_REG
                | ebpf::JSGE64_IMM
                | ebpf::JSGE64_REG
                | ebpf::JSLT64_IMM
                | ebpf::JSLT64_REG
                | ebpf::JSLE64_IMM
                | ebpf::JSLE64_REG
                | ebpf::JSET64_IMM
                | ebpf::JSET64_REG
                | ebpf::JEQ32_IMM
                | ebpf::JEQ32_REG
                | ebpf::JNE32_IMM
                | ebpf::JNE32_REG
                | ebpf::JGT32_IMM
                | ebpf::JGT32_REG
                | ebpf::JGE32_IMM
                | ebpf::JGE32_REG
                | ebpf::JLT32_IMM
                | ebpf::JLT32_REG
                | ebpf::JLE32_IMM
                | ebpf::JLE32_REG
                | ebpf::JSGT32_IMM
                | ebpf::JSGT32_REG
                | ebpf::JSGE32_IMM
                | ebpf::JSGE32_REG
                | ebpf::JSLT32_IMM
                | ebpf::JSLT32_REG
                | ebpf::JSLE32_IMM
                | ebpf::JSLE32_REG
                | ebpf::JSET32_IMM
                | ebpf::JSET32_REG
        )
    }

    /// Classify a terminator instruction.
    fn classify_terminator(&self, opcode: u8, _insn: &Insn) -> BlockTerminator {
        match opcode {
            ebpf::JA => BlockTerminator::UnconditionalJump,
            ebpf::EXIT => BlockTerminator::Exit,
            ebpf::CALL_IMM => BlockTerminator::FunctionCall { is_indirect: false },
            ebpf::CALL_REG => BlockTerminator::FunctionCall { is_indirect: true },
            _ if Self::is_conditional_branch(opcode) => {
                let (condition, is_immediate, is_32bit) =
                    Self::get_branch_condition(opcode).unwrap_or((BranchCondition::Equal, true, false));
                BlockTerminator::ConditionalBranch {
                    condition,
                    is_immediate,
                    is_32bit,
                }
            }
            _ => BlockTerminator::FallThrough,
        }
    }

    /// Get branch condition info from opcode.
    fn get_branch_condition(opcode: u8) -> Option<(BranchCondition, bool, bool)> {
        // Returns (condition, is_immediate, is_32bit)
        match opcode {
            ebpf::JEQ64_IMM => Some((BranchCondition::Equal, true, false)),
            ebpf::JEQ64_REG => Some((BranchCondition::Equal, false, false)),
            ebpf::JNE64_IMM => Some((BranchCondition::NotEqual, true, false)),
            ebpf::JNE64_REG => Some((BranchCondition::NotEqual, false, false)),
            ebpf::JGT64_IMM => Some((BranchCondition::GreaterThan, true, false)),
            ebpf::JGT64_REG => Some((BranchCondition::GreaterThan, false, false)),
            ebpf::JGE64_IMM => Some((BranchCondition::GreaterOrEqual, true, false)),
            ebpf::JGE64_REG => Some((BranchCondition::GreaterOrEqual, false, false)),
            ebpf::JLT64_IMM => Some((BranchCondition::LessThan, true, false)),
            ebpf::JLT64_REG => Some((BranchCondition::LessThan, false, false)),
            ebpf::JLE64_IMM => Some((BranchCondition::LessOrEqual, true, false)),
            ebpf::JLE64_REG => Some((BranchCondition::LessOrEqual, false, false)),
            ebpf::JSGT64_IMM => Some((BranchCondition::SignedGreaterThan, true, false)),
            ebpf::JSGT64_REG => Some((BranchCondition::SignedGreaterThan, false, false)),
            ebpf::JSGE64_IMM => Some((BranchCondition::SignedGreaterOrEqual, true, false)),
            ebpf::JSGE64_REG => Some((BranchCondition::SignedGreaterOrEqual, false, false)),
            ebpf::JSLT64_IMM => Some((BranchCondition::SignedLessThan, true, false)),
            ebpf::JSLT64_REG => Some((BranchCondition::SignedLessThan, false, false)),
            ebpf::JSLE64_IMM => Some((BranchCondition::SignedLessOrEqual, true, false)),
            ebpf::JSLE64_REG => Some((BranchCondition::SignedLessOrEqual, false, false)),
            ebpf::JSET64_IMM => Some((BranchCondition::BitTest, true, false)),
            ebpf::JSET64_REG => Some((BranchCondition::BitTest, false, false)),
            // 32-bit variants
            ebpf::JEQ32_IMM => Some((BranchCondition::Equal, true, true)),
            ebpf::JEQ32_REG => Some((BranchCondition::Equal, false, true)),
            ebpf::JNE32_IMM => Some((BranchCondition::NotEqual, true, true)),
            ebpf::JNE32_REG => Some((BranchCondition::NotEqual, false, true)),
            ebpf::JGT32_IMM => Some((BranchCondition::GreaterThan, true, true)),
            ebpf::JGT32_REG => Some((BranchCondition::GreaterThan, false, true)),
            ebpf::JGE32_IMM => Some((BranchCondition::GreaterOrEqual, true, true)),
            ebpf::JGE32_REG => Some((BranchCondition::GreaterOrEqual, false, true)),
            ebpf::JLT32_IMM => Some((BranchCondition::LessThan, true, true)),
            ebpf::JLT32_REG => Some((BranchCondition::LessThan, false, true)),
            ebpf::JLE32_IMM => Some((BranchCondition::LessOrEqual, true, true)),
            ebpf::JLE32_REG => Some((BranchCondition::LessOrEqual, false, true)),
            ebpf::JSGT32_IMM => Some((BranchCondition::SignedGreaterThan, true, true)),
            ebpf::JSGT32_REG => Some((BranchCondition::SignedGreaterThan, false, true)),
            ebpf::JSGE32_IMM => Some((BranchCondition::SignedGreaterOrEqual, true, true)),
            ebpf::JSGE32_REG => Some((BranchCondition::SignedGreaterOrEqual, false, true)),
            ebpf::JSLT32_IMM => Some((BranchCondition::SignedLessThan, true, true)),
            ebpf::JSLT32_REG => Some((BranchCondition::SignedLessThan, false, true)),
            ebpf::JSLE32_IMM => Some((BranchCondition::SignedLessOrEqual, true, true)),
            ebpf::JSLE32_REG => Some((BranchCondition::SignedLessOrEqual, false, true)),
            ebpf::JSET32_IMM => Some((BranchCondition::BitTest, true, true)),
            ebpf::JSET32_REG => Some((BranchCondition::BitTest, false, true)),
            _ => None,
        }
    }

    /// Get the branch target PC.
    fn get_branch_target(&self, pc: u64, insn: &Insn, opcode: u8) -> Option<u64> {
        match opcode {
            ebpf::JA => Some((pc as i64 + 1 + insn.off as i64) as u64),
            ebpf::EXIT => None,
            ebpf::CALL_IMM => {
                // Internal call: target = pc + 1 + imm
                if insn.src == 1 {
                    Some((pc as i64 + 1 + insn.imm as i64) as u64)
                } else {
                    None // Syscall
                }
            }
            ebpf::CALL_REG => None, // Indirect, target unknown statically
            _ if Self::is_conditional_branch(opcode) => {
                Some((pc as i64 + 1 + insn.off as i64) as u64)
            }
            _ => None,
        }
    }

    /// Record branch info for condition inference.
    fn record_branch_info(
        &mut self,
        opcode: u8,
        insn: &Insn,
        pre_regs: &[u64; 12],
        post_regs: &[u64; 12],
    ) {
        let (condition, is_immediate, is_32bit) = match Self::get_branch_condition(opcode) {
            Some(c) => c,
            None => return,
        };

        let dst_value = if is_32bit {
            pre_regs[insn.dst as usize] as u32 as u64
        } else {
            pre_regs[insn.dst as usize]
        };

        let comparand = if is_immediate {
            BranchComparand::Immediate { value: insn.imm }
        } else {
            let src_value = if is_32bit {
                pre_regs[insn.src as usize] as u32 as u64
            } else {
                pre_regs[insn.src as usize]
            };
            BranchComparand::Register {
                reg: insn.src,
                value: src_value,
            }
        };

        // Determine if branch was taken by comparing actual next PC to fall-through
        let expected_fallthrough = insn.ptr as u64 + 1;
        let was_taken = post_regs[11] != expected_fallthrough;

        self.branch_history.push(BranchTakenInfo {
            condition,
            dst_register: insn.dst,
            dst_value,
            comparand,
            was_taken,
        });
    }

    /// Record an edge in the CFG.
    fn record_edge(
        &mut self,
        from_pc: u64,
        to_pc: u64,
        opcode: u8,
        _insn: &Insn,
        _pre_regs: &[u64; 12],
        _post_regs: &[u64; 12],
    ) {
        let from_block = self.current_block_start.unwrap_or(from_pc);

        // Determine edge type
        let edge_type = if opcode == ebpf::EXIT {
            EdgeType::Return
        } else if opcode == ebpf::CALL_IMM || opcode == ebpf::CALL_REG {
            EdgeType::Call
        } else if Self::is_conditional_branch(opcode) {
            let expected_fallthrough = from_pc + 1;
            if to_pc == expected_fallthrough {
                EdgeType::ConditionalFallthrough
            } else if to_pc <= from_pc {
                EdgeType::BackEdge
            } else {
                EdgeType::ConditionalTaken
            }
        } else if to_pc <= from_pc {
            EdgeType::BackEdge
        } else {
            EdgeType::Unconditional
        };

        // Get branch info if conditional
        let branch_info = if Self::is_conditional_branch(opcode) {
            self.branch_history.last().cloned()
        } else {
            None
        };

        let key = (from_block, to_pc);
        let entry = self.edge_counts.entry(key).or_insert((0, edge_type, branch_info.clone()));
        entry.0 += 1;
    }

    /// Finalize the current block.
    fn finalize_current_block(&mut self, terminator: BlockTerminator) {
        if let Some(start_pc) = self.current_block_start.take() {
            let end_pc = *self.current_block_pcs.last().unwrap_or(&start_pc);
            let instruction_count = self.current_block_pcs.len() as u32;

            // Get or create block
            let label = self.symbol_map.get(&start_pc).cloned();
            let block = self.blocks.entry(start_pc).or_insert_with(|| BasicBlock {
                start_pc,
                end_pc,
                instruction_count,
                terminator: terminator.clone(),
                label,
                execution_count: 0,
            });

            block.execution_count += 1;
            block.terminator = terminator;
            block.end_pc = end_pc;
            block.instruction_count = instruction_count;

            self.current_block_pcs.clear();
        }
    }

    /// Detect loops from back edges.
    fn detect_loops(&self) -> Vec<DetectedLoop> {
        let mut loops = Vec::new();

        for &(back_edge_pc, header_pc) in &self.back_edges {
            // Find all blocks between header and back edge
            let body_blocks: Vec<u64> = self
                .blocks
                .keys()
                .filter(|&&pc| pc >= header_pc && pc <= back_edge_pc)
                .copied()
                .collect();

            let iteration_count = self.loop_iterations.get(&header_pc).copied().unwrap_or(1);

            // Try to infer loop bounds from branch history
            let estimated_bound = self.infer_loop_bounds(header_pc, iteration_count);

            // Avoid duplicate loops
            if !loops.iter().any(|l: &DetectedLoop| l.header_pc == header_pc) {
                loops.push(DetectedLoop {
                    header_pc,
                    back_edge_pc,
                    body_blocks,
                    iteration_count,
                    estimated_bound,
                });
            }
        }

        loops
    }

    /// Try to infer loop bounds from branch history.
    fn infer_loop_bounds(&self, _header_pc: u64, iteration_count: u32) -> Option<LoopBound> {
        if self.branch_history.len() < 2 || iteration_count < 2 {
            return None;
        }

        // Look for a register that changes predictably
        let first = self.branch_history.first()?;
        let last = self.branch_history.last()?;

        if first.dst_register != last.dst_register {
            return None;
        }

        let iterations = iteration_count as i64;
        if iterations == 0 {
            return None;
        }

        let step = (last.dst_value as i64 - first.dst_value as i64) / iterations;

        if step == 0 {
            return None;
        }

        Some(LoopBound {
            counter_register: first.dst_register,
            initial_value: first.dst_value,
            final_value: last.dst_value,
            step,
        })
    }

    /// Build the final CFG.
    pub fn build(mut self) -> ControlFlowGraph {
        // Finalize any remaining block
        if self.current_block_start.is_some() {
            self.finalize_current_block(BlockTerminator::FallThrough);
        }

        // Detect loops before consuming edge_counts
        let loops = self.detect_loops();

        // Build edges (consumes edge_counts)
        let edges: Vec<CfgEdge> = self
            .edge_counts
            .into_iter()
            .map(|((from, to), (count, edge_type, branch_info))| CfgEdge {
                from_block: from,
                to_block: to,
                edge_type,
                execution_count: count,
                branch_info,
            })
            .collect();

        // Find exit PCs
        let exit_pcs: Vec<u64> = self
            .blocks
            .iter()
            .filter(|(_, b)| b.terminator == BlockTerminator::Exit)
            .map(|(pc, _)| *pc)
            .collect();

        ControlFlowGraph {
            basic_blocks: self.blocks,
            edges,
            loops,
            entry_pc: self.entry_pc,
            exit_pcs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branch_condition_classification() {
        assert_eq!(
            CfgBuilder::get_branch_condition(ebpf::JEQ64_IMM),
            Some((BranchCondition::Equal, true, false))
        );
        assert_eq!(
            CfgBuilder::get_branch_condition(ebpf::JNE32_REG),
            Some((BranchCondition::NotEqual, false, true))
        );
        assert_eq!(
            CfgBuilder::get_branch_condition(ebpf::JSGT64_IMM),
            Some((BranchCondition::SignedGreaterThan, true, false))
        );
    }

    #[test]
    fn test_is_block_terminator() {
        assert!(CfgBuilder::is_block_terminator(ebpf::JA));
        assert!(CfgBuilder::is_block_terminator(ebpf::JEQ64_IMM));
        assert!(CfgBuilder::is_block_terminator(ebpf::EXIT));
        assert!(CfgBuilder::is_block_terminator(ebpf::CALL_IMM));
        assert!(!CfgBuilder::is_block_terminator(ebpf::ADD64_IMM));
        assert!(!CfgBuilder::is_block_terminator(ebpf::MOV64_REG));
    }

    #[test]
    fn test_is_conditional_branch() {
        assert!(CfgBuilder::is_conditional_branch(ebpf::JEQ64_IMM));
        assert!(CfgBuilder::is_conditional_branch(ebpf::JNE32_REG));
        assert!(!CfgBuilder::is_conditional_branch(ebpf::JA));
        assert!(!CfgBuilder::is_conditional_branch(ebpf::EXIT));
    }

    #[test]
    fn test_cfg_builder_new() {
        let builder = CfgBuilder::new(HashMap::new());
        assert!(builder.blocks.is_empty());
        assert!(builder.jump_targets.is_empty());
    }
}
