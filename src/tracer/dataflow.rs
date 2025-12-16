//! Data Flow Analysis for dynamic execution traces.
//!
//! This module provides runtime tracking of:
//! - Value lineage (where each value came from)
//! - Def-use chains (linking uses to definitions)
//! - Taint analysis (input propagation tracking)
//! - Memory alias tracking (stack store/load correlation)

use crate::ebpf;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::MemoryRegionType;

/// Unique identifier for a value definition point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DefId {
    /// Program counter where the value was defined.
    pub pc: u64,
    /// Sequence number for multiple definitions at same PC (e.g., loop iterations).
    pub seq: u64,
}

impl DefId {
    /// Create a new DefId.
    pub fn new(pc: u64, seq: u64) -> Self {
        Self { pc, seq }
    }

    /// Special DefId for function entry parameters.
    pub fn entry_param(reg: u8) -> Self {
        Self {
            pc: u64::MAX,
            seq: reg as u64,
        }
    }

    /// Check if this is an entry parameter.
    pub fn is_entry_param(&self) -> bool {
        self.pc == u64::MAX
    }
}

impl std::fmt::Display for DefId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_entry_param() {
            write!(f, "entry:r{}", self.seq)
        } else {
            write!(f, "{}:{}", self.pc, self.seq)
        }
    }
}

/// The origin/source of a value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ValueOrigin {
    /// Value is a compile-time constant (immediate).
    Constant { value: u64 },
    /// Value loaded from memory.
    MemoryLoad {
        address: u64,
        size: u64,
        /// The definition that stored this value (if known).
        #[serde(skip_serializing_if = "Option::is_none")]
        store_def: Option<DefId>,
    },
    /// Value copied from another register.
    RegisterCopy { source_reg: u8, source_def: DefId },
    /// Value computed from an operation.
    Computed {
        operation: OperationType,
        /// Input definition IDs used in computation.
        inputs: Vec<DefId>,
    },
    /// Value from function entry (r1-r5 are arguments, r10 is stack pointer).
    EntryParameter { register: u8 },
    /// Value from syscall return.
    SyscallReturn {
        syscall_name: String,
        /// Definitions of arguments passed to syscall.
        arg_defs: Vec<DefId>,
    },
    /// Value from function return.
    FunctionReturn {
        target_pc: u64,
        /// Definitions of arguments passed to function.
        arg_defs: Vec<DefId>,
    },
    /// Unknown origin (fallback).
    Unknown,
}

/// Types of computational operations that produce values.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OperationType {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Or,
    And,
    Xor,
    Lsh,
    Rsh,
    Arsh,
    Neg,
    Endian,
}

impl OperationType {
    /// Get operation type from opcode.
    pub fn from_opcode(opcode: u8) -> Option<Self> {
        match opcode {
            ebpf::ADD32_IMM | ebpf::ADD32_REG | ebpf::ADD64_IMM | ebpf::ADD64_REG => {
                Some(Self::Add)
            }
            ebpf::SUB32_IMM | ebpf::SUB32_REG | ebpf::SUB64_IMM | ebpf::SUB64_REG => {
                Some(Self::Sub)
            }
            ebpf::MUL32_IMM | ebpf::MUL32_REG | ebpf::MUL64_IMM | ebpf::MUL64_REG => {
                Some(Self::Mul)
            }
            ebpf::DIV32_IMM | ebpf::DIV32_REG | ebpf::DIV64_IMM | ebpf::DIV64_REG => {
                Some(Self::Div)
            }
            ebpf::MOD32_IMM | ebpf::MOD32_REG | ebpf::MOD64_IMM | ebpf::MOD64_REG => {
                Some(Self::Mod)
            }
            ebpf::OR32_IMM | ebpf::OR32_REG | ebpf::OR64_IMM | ebpf::OR64_REG => Some(Self::Or),
            ebpf::AND32_IMM | ebpf::AND32_REG | ebpf::AND64_IMM | ebpf::AND64_REG => {
                Some(Self::And)
            }
            ebpf::XOR32_IMM | ebpf::XOR32_REG | ebpf::XOR64_IMM | ebpf::XOR64_REG => {
                Some(Self::Xor)
            }
            ebpf::LSH32_IMM | ebpf::LSH32_REG | ebpf::LSH64_IMM | ebpf::LSH64_REG => {
                Some(Self::Lsh)
            }
            ebpf::RSH32_IMM | ebpf::RSH32_REG | ebpf::RSH64_IMM | ebpf::RSH64_REG => {
                Some(Self::Rsh)
            }
            ebpf::ARSH32_IMM | ebpf::ARSH32_REG | ebpf::ARSH64_IMM | ebpf::ARSH64_REG => {
                Some(Self::Arsh)
            }
            ebpf::NEG32 | ebpf::NEG64 => Some(Self::Neg),
            ebpf::LE | ebpf::BE => Some(Self::Endian),
            _ => None,
        }
    }
}

/// Taint label indicating data provenance.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TaintLabel {
    /// Value derived from function argument register at entry.
    InputArg { register: u8 },
    /// Value derived from memory load.
    MemoryInput { region: MemoryRegionType },
    /// Value derived from syscall return.
    SyscallResult { syscall_name: String },
    /// Value is a known constant (no input taint).
    Constant,
}

/// Complete information about a value definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueDefinition {
    /// Unique identifier for this definition.
    pub def_id: DefId,
    /// The actual value (if captured).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<u64>,
    /// How this value was produced.
    pub origin: ValueOrigin,
    /// Taint labels propagated to this value.
    #[serde(skip_serializing_if = "HashSet::is_empty")]
    pub taint: HashSet<TaintLabel>,
    /// Destination register or memory location.
    pub destination: ValueLocation,
}

/// Where a value is stored.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ValueLocation {
    Register { reg: u8 },
    Memory { address: u64, size: u64 },
}

/// A use of a value (reference to its definition).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueUse {
    /// PC where the value was used.
    pub use_pc: u64,
    /// The definition being used.
    pub def_id: DefId,
    /// What the value was used for.
    pub use_type: UseType,
}

/// How a value is used in an instruction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum UseType {
    /// Used as operand in computation.
    Operand,
    /// Used as address for memory access.
    Address,
    /// Used in conditional branch comparison.
    Condition,
    /// Used as function call target (indirect call).
    CallTarget,
    /// Used as syscall/function argument.
    Argument { position: u8 },
    /// Used as return value.
    ReturnValue,
}

/// Tracks a memory store for alias analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStore {
    /// Definition ID of the store instruction.
    pub store_def: DefId,
    /// Memory address written to.
    pub address: u64,
    /// Size of the store.
    pub size: u64,
    /// Definition ID of the value stored.
    pub value_def: DefId,
    /// Actual value stored.
    pub value: u64,
}

/// Complete data flow analysis state and results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DataFlowState {
    /// Current sequence counter (for DefId generation).
    #[serde(skip)]
    pub sequence: u64,
    /// Current definition for each register (r0-r10).
    #[serde(skip)]
    pub register_defs: [Option<DefId>; 11],
    /// All value definitions captured.
    #[serde(serialize_with = "serialize_def_map", deserialize_with = "deserialize_def_map")]
    pub definitions: HashMap<DefId, ValueDefinition>,
    /// All value uses captured.
    pub uses: Vec<ValueUse>,
    /// Memory stores (for alias tracking) - keyed by address.
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub memory_stores: HashMap<u64, Vec<MemoryStore>>,
    /// Def-use chains: definition -> list of uses.
    #[serde(
        skip_serializing_if = "HashMap::is_empty",
        serialize_with = "serialize_def_use_map",
        deserialize_with = "deserialize_def_use_map",
        default
    )]
    pub def_use_chains: HashMap<DefId, Vec<ValueUse>>,
}

/// Serialize HashMap<DefId, ValueDefinition> with string keys.
fn serialize_def_map<S>(
    map: &HashMap<DefId, ValueDefinition>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeMap;
    let mut ser_map = serializer.serialize_map(Some(map.len()))?;
    for (k, v) in map {
        ser_map.serialize_entry(&k.to_string(), v)?;
    }
    ser_map.end()
}

/// Deserialize HashMap<DefId, ValueDefinition> from string keys.
fn deserialize_def_map<'de, D>(
    deserializer: D,
) -> Result<HashMap<DefId, ValueDefinition>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{MapAccess, Visitor};

    struct DefMapVisitor;

    impl<'de> Visitor<'de> for DefMapVisitor {
        type Value = HashMap<DefId, ValueDefinition>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a map with DefId string keys")
        }

        fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let mut map = HashMap::new();
            while let Some((key, value)) = access.next_entry::<String, ValueDefinition>()? {
                // Parse "pc:seq" or "entry:rN" format
                let def_id = parse_def_id(&key).map_err(serde::de::Error::custom)?;
                map.insert(def_id, value);
            }
            Ok(map)
        }
    }

    deserializer.deserialize_map(DefMapVisitor)
}

/// Serialize HashMap<DefId, Vec<ValueUse>> with string keys.
fn serialize_def_use_map<S>(
    map: &HashMap<DefId, Vec<ValueUse>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeMap;
    let mut ser_map = serializer.serialize_map(Some(map.len()))?;
    for (k, v) in map {
        ser_map.serialize_entry(&k.to_string(), v)?;
    }
    ser_map.end()
}

/// Deserialize HashMap<DefId, Vec<ValueUse>> from string keys.
fn deserialize_def_use_map<'de, D>(
    deserializer: D,
) -> Result<HashMap<DefId, Vec<ValueUse>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{MapAccess, Visitor};

    struct DefUseMapVisitor;

    impl<'de> Visitor<'de> for DefUseMapVisitor {
        type Value = HashMap<DefId, Vec<ValueUse>>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a map with DefId string keys")
        }

        fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let mut map = HashMap::new();
            while let Some((key, value)) = access.next_entry::<String, Vec<ValueUse>>()? {
                let def_id = parse_def_id(&key).map_err(serde::de::Error::custom)?;
                map.insert(def_id, value);
            }
            Ok(map)
        }
    }

    deserializer.deserialize_map(DefUseMapVisitor)
}

/// Parse a DefId from its string representation.
fn parse_def_id(s: &str) -> Result<DefId, String> {
    if s.starts_with("entry:r") {
        let reg_str = &s[7..];
        let reg: u64 = reg_str.parse().map_err(|_| format!("Invalid entry param: {}", s))?;
        Ok(DefId::entry_param(reg as u8))
    } else {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid DefId format: {}", s));
        }
        let pc: u64 = parts[0].parse().map_err(|_| format!("Invalid pc in DefId: {}", s))?;
        let seq: u64 = parts[1].parse().map_err(|_| format!("Invalid seq in DefId: {}", s))?;
        Ok(DefId::new(pc, seq))
    }
}

impl DataFlowState {
    /// Create a new empty state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize with function entry parameters.
    pub fn init_entry_params(&mut self, initial_regs: &[u64; 12]) {
        // r1-r5 are argument registers (tainted as inputs)
        for reg in 1..=5u8 {
            let def_id = DefId::entry_param(reg);
            let mut taint = HashSet::new();
            taint.insert(TaintLabel::InputArg { register: reg });
            let def = ValueDefinition {
                def_id,
                value: Some(initial_regs[reg as usize]),
                origin: ValueOrigin::EntryParameter { register: reg },
                taint,
                destination: ValueLocation::Register { reg },
            };
            self.definitions.insert(def_id, def);
            self.register_defs[reg as usize] = Some(def_id);
        }

        // r10 is stack pointer (not input-tainted)
        let r10_def_id = DefId::entry_param(10);
        let r10_def = ValueDefinition {
            def_id: r10_def_id,
            value: Some(initial_regs[10]),
            origin: ValueOrigin::EntryParameter { register: 10 },
            taint: HashSet::new(),
            destination: ValueLocation::Register { reg: 10 },
        };
        self.definitions.insert(r10_def_id, r10_def);
        self.register_defs[10] = Some(r10_def_id);
    }

    /// Generate next DefId for given PC.
    pub fn next_def_id(&mut self, pc: u64) -> DefId {
        let def_id = DefId::new(pc, self.sequence);
        self.sequence += 1;
        def_id
    }

    /// Get current definition for a register.
    pub fn get_reg_def(&self, reg: u8) -> Option<DefId> {
        self.register_defs.get(reg as usize).copied().flatten()
    }

    /// Record a new value definition.
    pub fn define_value(&mut self, def: ValueDefinition) {
        let def_id = def.def_id;
        match &def.destination {
            ValueLocation::Register { reg } => {
                self.register_defs[*reg as usize] = Some(def_id);
            }
            ValueLocation::Memory { address, size } => {
                let store = MemoryStore {
                    store_def: def_id,
                    address: *address,
                    size: *size,
                    value_def: def_id,
                    value: def.value.unwrap_or(0),
                };
                self.memory_stores.entry(*address).or_default().push(store);
            }
        }
        self.definitions.insert(def_id, def);
    }

    /// Record a value use.
    pub fn record_use(&mut self, use_record: ValueUse) {
        let def_id = use_record.def_id;
        self.uses.push(use_record.clone());
        self.def_use_chains
            .entry(def_id)
            .or_default()
            .push(use_record);
    }

    /// Find the most recent store to an address.
    pub fn find_store_at(&self, address: u64, size: u64) -> Option<&MemoryStore> {
        if let Some(stores) = self.memory_stores.get(&address) {
            for store in stores.iter().rev() {
                if store.size >= size {
                    return Some(store);
                }
            }
        }
        None
    }

    /// Compute combined taint from multiple definitions.
    pub fn combine_taint(&self, def_ids: &[DefId]) -> HashSet<TaintLabel> {
        let mut combined = HashSet::new();
        for def_id in def_ids {
            if let Some(def) = self.definitions.get(def_id) {
                combined.extend(def.taint.iter().cloned());
            }
        }
        combined
    }

    /// Finalize analysis and build def-use chains.
    pub fn finalize(&mut self) {
        self.def_use_chains.clear();
        for use_record in &self.uses {
            self.def_use_chains
                .entry(use_record.def_id)
                .or_default()
                .push(use_record.clone());
        }
    }
}

/// Analyzer that processes instructions and updates data flow state.
pub struct DataFlowAnalyzer {
    /// Current analysis state.
    pub state: DataFlowState,
    /// Whether to track all definitions or only tainted ones.
    pub track_all: bool,
}

impl DataFlowAnalyzer {
    /// Create a new analyzer.
    pub fn new(track_all: bool) -> Self {
        Self {
            state: DataFlowState::new(),
            track_all,
        }
    }

    /// Initialize with entry point registers.
    pub fn init(&mut self, initial_regs: &[u64; 12]) {
        self.state.init_entry_params(initial_regs);
    }

    /// Analyze an instruction and update data flow state.
    pub fn analyze_instruction(
        &mut self,
        pc: u64,
        opcode: u8,
        insn: &ebpf::Insn,
        pre_regs: &[u64; 12],
        post_regs: &[u64; 12],
        memory_region: Option<MemoryRegionType>,
        mem_address: Option<u64>,
        mem_size: Option<u64>,
    ) -> Option<ValueDefinition> {
        match opcode {
            // MOV immediate
            ebpf::MOV32_IMM | ebpf::MOV64_IMM => {
                self.handle_mov_imm(pc, insn, post_regs)
            }

            // MOV register
            ebpf::MOV32_REG | ebpf::MOV64_REG => {
                self.handle_mov_reg(pc, insn, pre_regs, post_regs)
            }

            // ALU operations with immediate
            ebpf::ADD32_IMM | ebpf::SUB32_IMM | ebpf::MUL32_IMM | ebpf::DIV32_IMM |
            ebpf::OR32_IMM | ebpf::AND32_IMM | ebpf::XOR32_IMM | ebpf::LSH32_IMM |
            ebpf::RSH32_IMM | ebpf::MOD32_IMM | ebpf::ARSH32_IMM |
            ebpf::ADD64_IMM | ebpf::SUB64_IMM | ebpf::MUL64_IMM | ebpf::DIV64_IMM |
            ebpf::OR64_IMM | ebpf::AND64_IMM | ebpf::XOR64_IMM | ebpf::LSH64_IMM |
            ebpf::RSH64_IMM | ebpf::MOD64_IMM | ebpf::ARSH64_IMM | ebpf::HOR64_IMM => {
                self.handle_alu_imm(pc, opcode, insn, post_regs)
            }

            // ALU operations with register
            ebpf::ADD32_REG | ebpf::SUB32_REG | ebpf::MUL32_REG | ebpf::DIV32_REG |
            ebpf::OR32_REG | ebpf::AND32_REG | ebpf::XOR32_REG | ebpf::LSH32_REG |
            ebpf::RSH32_REG | ebpf::MOD32_REG | ebpf::ARSH32_REG |
            ebpf::ADD64_REG | ebpf::SUB64_REG | ebpf::MUL64_REG | ebpf::DIV64_REG |
            ebpf::OR64_REG | ebpf::AND64_REG | ebpf::XOR64_REG | ebpf::LSH64_REG |
            ebpf::RSH64_REG | ebpf::MOD64_REG | ebpf::ARSH64_REG => {
                self.handle_alu_reg(pc, opcode, insn, post_regs)
            }

            // Unary operations
            ebpf::NEG32 | ebpf::NEG64 => {
                self.handle_neg(pc, opcode, insn, post_regs)
            }

            // Endianness conversion
            ebpf::LE | ebpf::BE => {
                self.handle_endian(pc, insn, post_regs)
            }

            // Load double-word immediate
            ebpf::LD_DW_IMM => {
                self.handle_lddw(pc, insn, post_regs)
            }

            // Memory loads
            // Note: LD_*B_REG opcodes share values with ALU32 opcodes, handled above
            ebpf::LD_B_REG | ebpf::LD_H_REG | ebpf::LD_W_REG | ebpf::LD_DW_REG => {
                self.handle_load(pc, insn, post_regs, memory_region, mem_address, mem_size)
            }

            // Memory stores
            // Note: ST_*B_* opcodes share values with ALU64 opcodes, handled above
            ebpf::ST_B_IMM | ebpf::ST_H_IMM | ebpf::ST_W_IMM | ebpf::ST_DW_IMM |
            ebpf::ST_B_REG | ebpf::ST_H_REG | ebpf::ST_W_REG | ebpf::ST_DW_REG => {
                self.handle_store(pc, opcode, insn, pre_regs, mem_address, mem_size)
            }

            // Conditional branches - record uses but no definition
            ebpf::JEQ32_IMM | ebpf::JGT32_IMM | ebpf::JGE32_IMM | ebpf::JLT32_IMM |
            ebpf::JLE32_IMM | ebpf::JSET32_IMM | ebpf::JNE32_IMM | ebpf::JSGT32_IMM |
            ebpf::JSGE32_IMM | ebpf::JSLT32_IMM | ebpf::JSLE32_IMM |
            ebpf::JEQ64_IMM | ebpf::JGT64_IMM | ebpf::JGE64_IMM | ebpf::JLT64_IMM |
            ebpf::JLE64_IMM | ebpf::JSET64_IMM | ebpf::JNE64_IMM | ebpf::JSGT64_IMM |
            ebpf::JSGE64_IMM | ebpf::JSLT64_IMM | ebpf::JSLE64_IMM => {
                self.record_condition_use_imm(pc, insn);
                None
            }

            ebpf::JEQ32_REG | ebpf::JGT32_REG | ebpf::JGE32_REG | ebpf::JLT32_REG |
            ebpf::JLE32_REG | ebpf::JSET32_REG | ebpf::JNE32_REG | ebpf::JSGT32_REG |
            ebpf::JSGE32_REG | ebpf::JSLT32_REG | ebpf::JSLE32_REG |
            ebpf::JEQ64_REG | ebpf::JGT64_REG | ebpf::JGE64_REG | ebpf::JLT64_REG |
            ebpf::JLE64_REG | ebpf::JSET64_REG | ebpf::JNE64_REG | ebpf::JSGT64_REG |
            ebpf::JSGE64_REG | ebpf::JSLT64_REG | ebpf::JSLE64_REG => {
                self.record_condition_use_reg(pc, insn);
                None
            }

            _ => None,
        }
    }

    fn handle_mov_imm(
        &mut self,
        pc: u64,
        insn: &ebpf::Insn,
        post_regs: &[u64; 12],
    ) -> Option<ValueDefinition> {
        let def_id = self.state.next_def_id(pc);
        let value = post_regs[insn.dst as usize];

        let def = ValueDefinition {
            def_id,
            value: Some(value),
            origin: ValueOrigin::Constant {
                value: insn.imm as u64,
            },
            taint: HashSet::from([TaintLabel::Constant]),
            destination: ValueLocation::Register { reg: insn.dst },
        };

        self.state.define_value(def.clone());
        Some(def)
    }

    fn handle_mov_reg(
        &mut self,
        pc: u64,
        insn: &ebpf::Insn,
        _pre_regs: &[u64; 12],
        post_regs: &[u64; 12],
    ) -> Option<ValueDefinition> {
        let def_id = self.state.next_def_id(pc);
        let src_def = self.state.get_reg_def(insn.src);
        let value = post_regs[insn.dst as usize];

        // Record use of source register
        if let Some(src_def_id) = src_def {
            self.state.record_use(ValueUse {
                use_pc: pc,
                def_id: src_def_id,
                use_type: UseType::Operand,
            });
        }

        // Get taint from source
        let taint = src_def
            .and_then(|d| self.state.definitions.get(&d))
            .map(|d| d.taint.clone())
            .unwrap_or_default();

        let origin = if let Some(src_def_id) = src_def {
            ValueOrigin::RegisterCopy {
                source_reg: insn.src,
                source_def: src_def_id,
            }
        } else {
            ValueOrigin::Unknown
        };

        let def = ValueDefinition {
            def_id,
            value: Some(value),
            origin,
            taint,
            destination: ValueLocation::Register { reg: insn.dst },
        };

        self.state.define_value(def.clone());
        Some(def)
    }

    fn handle_alu_imm(
        &mut self,
        pc: u64,
        opcode: u8,
        insn: &ebpf::Insn,
        post_regs: &[u64; 12],
    ) -> Option<ValueDefinition> {
        let def_id = self.state.next_def_id(pc);
        let dst_def = self.state.get_reg_def(insn.dst);
        let value = post_regs[insn.dst as usize];

        // Record use of dst register
        let mut inputs = Vec::new();
        if let Some(dst_def_id) = dst_def {
            self.state.record_use(ValueUse {
                use_pc: pc,
                def_id: dst_def_id,
                use_type: UseType::Operand,
            });
            inputs.push(dst_def_id);
        }

        let taint = self.state.combine_taint(&inputs);
        let operation = OperationType::from_opcode(opcode).unwrap_or(OperationType::Add);

        let def = ValueDefinition {
            def_id,
            value: Some(value),
            origin: ValueOrigin::Computed {
                operation,
                inputs,
            },
            taint,
            destination: ValueLocation::Register { reg: insn.dst },
        };

        self.state.define_value(def.clone());
        Some(def)
    }

    fn handle_alu_reg(
        &mut self,
        pc: u64,
        opcode: u8,
        insn: &ebpf::Insn,
        post_regs: &[u64; 12],
    ) -> Option<ValueDefinition> {
        let def_id = self.state.next_def_id(pc);
        let dst_def = self.state.get_reg_def(insn.dst);
        let src_def = self.state.get_reg_def(insn.src);
        let value = post_regs[insn.dst as usize];

        let mut inputs = Vec::new();
        if let Some(dst_def_id) = dst_def {
            self.state.record_use(ValueUse {
                use_pc: pc,
                def_id: dst_def_id,
                use_type: UseType::Operand,
            });
            inputs.push(dst_def_id);
        }
        if let Some(src_def_id) = src_def {
            self.state.record_use(ValueUse {
                use_pc: pc,
                def_id: src_def_id,
                use_type: UseType::Operand,
            });
            inputs.push(src_def_id);
        }

        let taint = self.state.combine_taint(&inputs);
        let operation = OperationType::from_opcode(opcode).unwrap_or(OperationType::Add);

        let def = ValueDefinition {
            def_id,
            value: Some(value),
            origin: ValueOrigin::Computed {
                operation,
                inputs,
            },
            taint,
            destination: ValueLocation::Register { reg: insn.dst },
        };

        self.state.define_value(def.clone());
        Some(def)
    }

    fn handle_neg(
        &mut self,
        pc: u64,
        opcode: u8,
        insn: &ebpf::Insn,
        post_regs: &[u64; 12],
    ) -> Option<ValueDefinition> {
        let def_id = self.state.next_def_id(pc);
        let dst_def = self.state.get_reg_def(insn.dst);
        let value = post_regs[insn.dst as usize];

        let mut inputs = Vec::new();
        if let Some(dst_def_id) = dst_def {
            self.state.record_use(ValueUse {
                use_pc: pc,
                def_id: dst_def_id,
                use_type: UseType::Operand,
            });
            inputs.push(dst_def_id);
        }

        let taint = self.state.combine_taint(&inputs);
        let operation = OperationType::from_opcode(opcode).unwrap_or(OperationType::Neg);

        let def = ValueDefinition {
            def_id,
            value: Some(value),
            origin: ValueOrigin::Computed {
                operation,
                inputs,
            },
            taint,
            destination: ValueLocation::Register { reg: insn.dst },
        };

        self.state.define_value(def.clone());
        Some(def)
    }

    fn handle_endian(
        &mut self,
        pc: u64,
        insn: &ebpf::Insn,
        post_regs: &[u64; 12],
    ) -> Option<ValueDefinition> {
        let def_id = self.state.next_def_id(pc);
        let dst_def = self.state.get_reg_def(insn.dst);
        let value = post_regs[insn.dst as usize];

        let mut inputs = Vec::new();
        if let Some(dst_def_id) = dst_def {
            self.state.record_use(ValueUse {
                use_pc: pc,
                def_id: dst_def_id,
                use_type: UseType::Operand,
            });
            inputs.push(dst_def_id);
        }

        let taint = self.state.combine_taint(&inputs);

        let def = ValueDefinition {
            def_id,
            value: Some(value),
            origin: ValueOrigin::Computed {
                operation: OperationType::Endian,
                inputs,
            },
            taint,
            destination: ValueLocation::Register { reg: insn.dst },
        };

        self.state.define_value(def.clone());
        Some(def)
    }

    fn handle_lddw(
        &mut self,
        pc: u64,
        insn: &ebpf::Insn,
        post_regs: &[u64; 12],
    ) -> Option<ValueDefinition> {
        let def_id = self.state.next_def_id(pc);
        let value = post_regs[insn.dst as usize];

        let def = ValueDefinition {
            def_id,
            value: Some(value),
            origin: ValueOrigin::Constant { value },
            taint: HashSet::from([TaintLabel::Constant]),
            destination: ValueLocation::Register { reg: insn.dst },
        };

        self.state.define_value(def.clone());
        Some(def)
    }

    fn handle_load(
        &mut self,
        pc: u64,
        insn: &ebpf::Insn,
        post_regs: &[u64; 12],
        memory_region: Option<MemoryRegionType>,
        mem_address: Option<u64>,
        mem_size: Option<u64>,
    ) -> Option<ValueDefinition> {
        let def_id = self.state.next_def_id(pc);
        let src_def = self.state.get_reg_def(insn.src);
        let value = post_regs[insn.dst as usize];

        // Record use of address register
        if let Some(src_def_id) = src_def {
            self.state.record_use(ValueUse {
                use_pc: pc,
                def_id: src_def_id,
                use_type: UseType::Address,
            });
        }

        let address = mem_address.unwrap_or(0);
        let size = mem_size.unwrap_or(8);

        // Check for memory alias
        let store_def = self.state.find_store_at(address, size).map(|s| s.value_def);

        // Determine taint
        let taint = if let Some(store_def_id) = store_def {
            self.state
                .definitions
                .get(&store_def_id)
                .map(|d| d.taint.clone())
                .unwrap_or_default()
        } else {
            let region = memory_region.unwrap_or(MemoryRegionType::Unknown);
            HashSet::from([TaintLabel::MemoryInput { region }])
        };

        let def = ValueDefinition {
            def_id,
            value: Some(value),
            origin: ValueOrigin::MemoryLoad {
                address,
                size,
                store_def,
            },
            taint,
            destination: ValueLocation::Register { reg: insn.dst },
        };

        self.state.define_value(def.clone());
        Some(def)
    }

    fn handle_store(
        &mut self,
        pc: u64,
        opcode: u8,
        insn: &ebpf::Insn,
        pre_regs: &[u64; 12],
        mem_address: Option<u64>,
        mem_size: Option<u64>,
    ) -> Option<ValueDefinition> {
        let def_id = self.state.next_def_id(pc);

        let address = mem_address.unwrap_or(0);
        let size = mem_size.unwrap_or(8);

        // Record use of address register
        let dst_def = self.state.get_reg_def(insn.dst);
        if let Some(dst_def_id) = dst_def {
            self.state.record_use(ValueUse {
                use_pc: pc,
                def_id: dst_def_id,
                use_type: UseType::Address,
            });
        }

        // Determine if immediate or register store
        let is_imm_store = matches!(
            opcode,
            ebpf::ST_B_IMM
                | ebpf::ST_H_IMM
                | ebpf::ST_W_IMM
                | ebpf::ST_DW_IMM
                | ebpf::ST_1B_IMM
                | ebpf::ST_2B_IMM
                | ebpf::ST_4B_IMM
                | ebpf::ST_8B_IMM
        );

        let (origin, taint, value) = if is_imm_store {
            (
                ValueOrigin::Constant {
                    value: insn.imm as u64,
                },
                HashSet::from([TaintLabel::Constant]),
                Some(insn.imm as u64),
            )
        } else {
            let src_def = self.state.get_reg_def(insn.src);
            if let Some(src_def_id) = src_def {
                self.state.record_use(ValueUse {
                    use_pc: pc,
                    def_id: src_def_id,
                    use_type: UseType::Operand,
                });
                let src_taint = self
                    .state
                    .definitions
                    .get(&src_def_id)
                    .map(|d| d.taint.clone())
                    .unwrap_or_default();
                let src_value = Some(pre_regs[insn.src as usize]);
                (
                    ValueOrigin::RegisterCopy {
                        source_reg: insn.src,
                        source_def: src_def_id,
                    },
                    src_taint,
                    src_value,
                )
            } else {
                (ValueOrigin::Unknown, HashSet::new(), None)
            }
        };

        let def = ValueDefinition {
            def_id,
            value,
            origin,
            taint,
            destination: ValueLocation::Memory { address, size },
        };

        self.state.define_value(def.clone());
        Some(def)
    }

    fn record_condition_use_imm(&mut self, pc: u64, insn: &ebpf::Insn) {
        if let Some(dst_def_id) = self.state.get_reg_def(insn.dst) {
            self.state.record_use(ValueUse {
                use_pc: pc,
                def_id: dst_def_id,
                use_type: UseType::Condition,
            });
        }
    }

    fn record_condition_use_reg(&mut self, pc: u64, insn: &ebpf::Insn) {
        if let Some(dst_def_id) = self.state.get_reg_def(insn.dst) {
            self.state.record_use(ValueUse {
                use_pc: pc,
                def_id: dst_def_id,
                use_type: UseType::Condition,
            });
        }
        if let Some(src_def_id) = self.state.get_reg_def(insn.src) {
            self.state.record_use(ValueUse {
                use_pc: pc,
                def_id: src_def_id,
                use_type: UseType::Condition,
            });
        }
    }

    /// Handle syscall - record argument uses and define return value.
    pub fn handle_syscall(
        &mut self,
        pc: u64,
        syscall_name: &str,
        post_regs: &[u64; 12],
    ) -> Option<ValueDefinition> {
        // Record uses of r1-r5 as arguments
        let mut arg_defs = Vec::new();
        for reg in 1..=5u8 {
            if let Some(def_id) = self.state.get_reg_def(reg) {
                self.state.record_use(ValueUse {
                    use_pc: pc,
                    def_id,
                    use_type: UseType::Argument { position: reg },
                });
                arg_defs.push(def_id);
            }
        }

        // Define r0 with syscall return
        let def_id = self.state.next_def_id(pc);
        let value = post_regs[0];

        let taint = HashSet::from([TaintLabel::SyscallResult {
            syscall_name: syscall_name.to_string(),
        }]);

        let def = ValueDefinition {
            def_id,
            value: Some(value),
            origin: ValueOrigin::SyscallReturn {
                syscall_name: syscall_name.to_string(),
                arg_defs,
            },
            taint,
            destination: ValueLocation::Register { reg: 0 },
        };

        self.state.define_value(def.clone());
        Some(def)
    }

    /// Handle function call - record argument uses.
    pub fn handle_function_call(&mut self, pc: u64) {
        for reg in 1..=5u8 {
            if let Some(def_id) = self.state.get_reg_def(reg) {
                self.state.record_use(ValueUse {
                    use_pc: pc,
                    def_id,
                    use_type: UseType::Argument { position: reg },
                });
            }
        }
    }

    /// Handle function return - define r0 with return value.
    pub fn handle_function_return(
        &mut self,
        pc: u64,
        target_pc: u64,
        post_regs: &[u64; 12],
        arg_defs: Vec<DefId>,
    ) -> Option<ValueDefinition> {
        let def_id = self.state.next_def_id(pc);
        let value = post_regs[0];

        let taint = self.state.combine_taint(&arg_defs);

        let def = ValueDefinition {
            def_id,
            value: Some(value),
            origin: ValueOrigin::FunctionReturn { target_pc, arg_defs },
            taint,
            destination: ValueLocation::Register { reg: 0 },
        };

        self.state.define_value(def.clone());
        Some(def)
    }

    /// Get final analysis state.
    pub fn finalize(mut self) -> DataFlowState {
        self.state.finalize();
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_def_id() {
        let def = DefId::new(10, 5);
        assert_eq!(def.pc, 10);
        assert_eq!(def.seq, 5);
        assert!(!def.is_entry_param());

        let entry = DefId::entry_param(3);
        assert!(entry.is_entry_param());
    }

    #[test]
    fn test_operation_type_from_opcode() {
        assert_eq!(
            OperationType::from_opcode(ebpf::ADD64_IMM),
            Some(OperationType::Add)
        );
        assert_eq!(
            OperationType::from_opcode(ebpf::SUB32_REG),
            Some(OperationType::Sub)
        );
        assert_eq!(
            OperationType::from_opcode(ebpf::XOR64_REG),
            Some(OperationType::Xor)
        );
        assert_eq!(OperationType::from_opcode(ebpf::EXIT), None);
    }

    #[test]
    fn test_dataflow_state_init() {
        let mut state = DataFlowState::new();
        let regs = [0, 100, 200, 300, 400, 500, 0, 0, 0, 0, 1000, 0];
        state.init_entry_params(&regs);

        // Check r1-r5 are defined with input taint
        for reg in 1..=5u8 {
            let def_id = state.get_reg_def(reg);
            assert!(def_id.is_some());
            let def = state.definitions.get(&def_id.unwrap()).unwrap();
            assert!(def.taint.contains(&TaintLabel::InputArg { register: reg }));
        }

        // Check r10 is defined without input taint
        let r10_def_id = state.get_reg_def(10);
        assert!(r10_def_id.is_some());
        let r10_def = state.definitions.get(&r10_def_id.unwrap()).unwrap();
        assert!(r10_def.taint.is_empty());
    }

    #[test]
    fn test_combine_taint() {
        let mut state = DataFlowState::new();

        let def1 = ValueDefinition {
            def_id: DefId::new(0, 0),
            value: Some(10),
            origin: ValueOrigin::Constant { value: 10 },
            taint: HashSet::from([TaintLabel::InputArg { register: 1 }]),
            destination: ValueLocation::Register { reg: 0 },
        };
        state.definitions.insert(def1.def_id, def1.clone());

        let def2 = ValueDefinition {
            def_id: DefId::new(1, 1),
            value: Some(20),
            origin: ValueOrigin::Constant { value: 20 },
            taint: HashSet::from([TaintLabel::InputArg { register: 2 }]),
            destination: ValueLocation::Register { reg: 1 },
        };
        state.definitions.insert(def2.def_id, def2.clone());

        let combined = state.combine_taint(&[def1.def_id, def2.def_id]);
        assert!(combined.contains(&TaintLabel::InputArg { register: 1 }));
        assert!(combined.contains(&TaintLabel::InputArg { register: 2 }));
    }
}
