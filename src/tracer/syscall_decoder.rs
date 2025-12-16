//! Syscall argument decoders for semantic tracing.
//!
//! This module provides a registry of decoders that can interpret syscall
//! arguments and convert them to human-readable representations.

use crate::memory_region::{AccessType, MemoryMapping};
use std::collections::HashMap;

/// Maximum length for decoded string values.
const MAX_STRING_LEN: usize = 256;

/// Trait for syscall argument decoding.
///
/// Implementations decode raw register values (r1-r5) into semantic
/// key-value pairs for trace output.
pub trait SyscallDecoder: Send + Sync {
    /// Returns the syscall name this decoder handles.
    fn name(&self) -> &'static str;

    /// Decode arguments from r1-r5, optionally dereferencing pointers.
    fn decode_args(
        &self,
        r1: u64,
        r2: u64,
        r3: u64,
        r4: u64,
        r5: u64,
        memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String>;
}

/// Registry of syscall decoders.
pub struct SyscallDecoderRegistry {
    decoders: HashMap<String, Box<dyn SyscallDecoder>>,
}

impl Default for SyscallDecoderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SyscallDecoderRegistry {
    /// Create a new registry with built-in Solana syscall decoders.
    pub fn new() -> Self {
        let mut registry = Self {
            decoders: HashMap::new(),
        };

        // Register built-in Solana syscall decoders
        registry.register(Box::new(SolLogDecoder));
        registry.register(Box::new(SolLog64Decoder));
        registry.register(Box::new(SolLogPubkeyDecoder));
        registry.register(Box::new(SolLogComputeUnitsDecoder));
        registry.register(Box::new(SolMemcpyDecoder));
        registry.register(Box::new(SolMemsetDecoder));
        registry.register(Box::new(SolMemcmpDecoder));
        registry.register(Box::new(SolMemmoveDecoder));
        registry.register(Box::new(SolAllocFreeDecoder));
        registry.register(Box::new(SolInvokeSignedDecoder));
        registry.register(Box::new(SolSetReturnDataDecoder));
        registry.register(Box::new(SolGetReturnDataDecoder));
        registry.register(Box::new(SolSha256Decoder));
        registry.register(Box::new(SolKeccak256Decoder));
        registry.register(Box::new(SolBlake3Decoder));

        registry
    }

    /// Register a custom syscall decoder.
    pub fn register(&mut self, decoder: Box<dyn SyscallDecoder>) {
        self.decoders.insert(decoder.name().to_string(), decoder);
    }

    /// Decode syscall arguments using the appropriate decoder.
    ///
    /// If no decoder is registered for the syscall, returns raw argument values.
    pub fn decode(
        &self,
        name: &str,
        args: [u64; 5],
        memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String> {
        if let Some(decoder) = self.decoders.get(name) {
            decoder.decode_args(args[0], args[1], args[2], args[3], args[4], memory_mapping)
        } else {
            // Fallback: return raw arguments
            let mut map = HashMap::new();
            for (i, arg) in args.iter().enumerate() {
                map.insert(format!("r{}", i + 1), format!("{:#x}", arg));
            }
            map
        }
    }
}

/// Helper function to read a string from VM memory.
fn read_string_from_memory(
    memory_mapping: &MemoryMapping,
    ptr: u64,
    len: u64,
) -> Option<String> {
    let actual_len = len.min(MAX_STRING_LEN as u64);
    if actual_len == 0 {
        return Some(String::new());
    }

    let result: Result<u64, _> = memory_mapping.map(AccessType::Load, ptr, actual_len).into();
    if let Ok(host_addr) = result {
        let slice = unsafe { std::slice::from_raw_parts(host_addr as *const u8, actual_len as usize) };
        // Find null terminator if present
        let end = slice.iter().position(|&c| c == 0).unwrap_or(slice.len());
        String::from_utf8_lossy(&slice[..end]).to_string().into()
    } else {
        None
    }
}

/// Helper function to read bytes from VM memory as hex.
fn read_bytes_as_hex(
    memory_mapping: &MemoryMapping,
    ptr: u64,
    len: u64,
    max_len: usize,
) -> Option<String> {
    let actual_len = len.min(max_len as u64);
    if actual_len == 0 {
        return Some(String::new());
    }

    let result: Result<u64, _> = memory_mapping.map(AccessType::Load, ptr, actual_len).into();
    if let Ok(host_addr) = result {
        let slice = unsafe { std::slice::from_raw_parts(host_addr as *const u8, actual_len as usize) };
        Some(hex::encode(slice))
    } else {
        None
    }
}

// ============================================================================
// Solana Syscall Decoder Implementations
// ============================================================================

/// Decoder for sol_log_ syscall (log a message).
pub struct SolLogDecoder;

impl SyscallDecoder for SolLogDecoder {
    fn name(&self) -> &'static str {
        "sol_log_"
    }

    fn decode_args(
        &self,
        r1: u64,
        r2: u64,
        _r3: u64,
        _r4: u64,
        _r5: u64,
        memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String> {
        let mut args = HashMap::new();
        args.insert("msg_ptr".to_string(), format!("{:#x}", r1));
        args.insert("msg_len".to_string(), format!("{}", r2));

        if let Some(msg) = read_string_from_memory(memory_mapping, r1, r2) {
            args.insert("message".to_string(), msg);
        }
        args
    }
}

/// Decoder for sol_log_64_ syscall (log 5 u64 values).
pub struct SolLog64Decoder;

impl SyscallDecoder for SolLog64Decoder {
    fn name(&self) -> &'static str {
        "sol_log_64_"
    }

    fn decode_args(
        &self,
        r1: u64,
        r2: u64,
        r3: u64,
        r4: u64,
        r5: u64,
        _memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String> {
        let mut args = HashMap::new();
        args.insert("arg1".to_string(), format!("{}", r1));
        args.insert("arg2".to_string(), format!("{}", r2));
        args.insert("arg3".to_string(), format!("{}", r3));
        args.insert("arg4".to_string(), format!("{}", r4));
        args.insert("arg5".to_string(), format!("{}", r5));
        args
    }
}

/// Decoder for sol_log_pubkey syscall.
pub struct SolLogPubkeyDecoder;

impl SyscallDecoder for SolLogPubkeyDecoder {
    fn name(&self) -> &'static str {
        "sol_log_pubkey"
    }

    fn decode_args(
        &self,
        r1: u64,
        _r2: u64,
        _r3: u64,
        _r4: u64,
        _r5: u64,
        memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String> {
        let mut args = HashMap::new();
        args.insert("pubkey_ptr".to_string(), format!("{:#x}", r1));

        // Pubkey is 32 bytes
        if let Some(hex) = read_bytes_as_hex(memory_mapping, r1, 32, 32) {
            args.insert("pubkey_hex".to_string(), hex);
        }
        args
    }
}

/// Decoder for sol_log_compute_units_ syscall.
pub struct SolLogComputeUnitsDecoder;

impl SyscallDecoder for SolLogComputeUnitsDecoder {
    fn name(&self) -> &'static str {
        "sol_log_compute_units_"
    }

    fn decode_args(
        &self,
        _r1: u64,
        _r2: u64,
        _r3: u64,
        _r4: u64,
        _r5: u64,
        _memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String> {
        HashMap::new() // No arguments
    }
}

/// Decoder for sol_memcpy_ syscall.
pub struct SolMemcpyDecoder;

impl SyscallDecoder for SolMemcpyDecoder {
    fn name(&self) -> &'static str {
        "sol_memcpy_"
    }

    fn decode_args(
        &self,
        r1: u64,
        r2: u64,
        r3: u64,
        _r4: u64,
        _r5: u64,
        _memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String> {
        let mut args = HashMap::new();
        args.insert("dst".to_string(), format!("{:#x}", r1));
        args.insert("src".to_string(), format!("{:#x}", r2));
        args.insert("len".to_string(), format!("{}", r3));
        args
    }
}

/// Decoder for sol_memset_ syscall.
pub struct SolMemsetDecoder;

impl SyscallDecoder for SolMemsetDecoder {
    fn name(&self) -> &'static str {
        "sol_memset_"
    }

    fn decode_args(
        &self,
        r1: u64,
        r2: u64,
        r3: u64,
        _r4: u64,
        _r5: u64,
        _memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String> {
        let mut args = HashMap::new();
        args.insert("ptr".to_string(), format!("{:#x}", r1));
        args.insert("byte".to_string(), format!("{:#x}", r2 as u8));
        args.insert("len".to_string(), format!("{}", r3));
        args
    }
}

/// Decoder for sol_memcmp_ syscall.
pub struct SolMemcmpDecoder;

impl SyscallDecoder for SolMemcmpDecoder {
    fn name(&self) -> &'static str {
        "sol_memcmp_"
    }

    fn decode_args(
        &self,
        r1: u64,
        r2: u64,
        r3: u64,
        r4: u64,
        _r5: u64,
        _memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String> {
        let mut args = HashMap::new();
        args.insert("ptr1".to_string(), format!("{:#x}", r1));
        args.insert("ptr2".to_string(), format!("{:#x}", r2));
        args.insert("len".to_string(), format!("{}", r3));
        args.insert("result_ptr".to_string(), format!("{:#x}", r4));
        args
    }
}

/// Decoder for sol_memmove_ syscall.
pub struct SolMemmoveDecoder;

impl SyscallDecoder for SolMemmoveDecoder {
    fn name(&self) -> &'static str {
        "sol_memmove_"
    }

    fn decode_args(
        &self,
        r1: u64,
        r2: u64,
        r3: u64,
        _r4: u64,
        _r5: u64,
        _memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String> {
        let mut args = HashMap::new();
        args.insert("dst".to_string(), format!("{:#x}", r1));
        args.insert("src".to_string(), format!("{:#x}", r2));
        args.insert("len".to_string(), format!("{}", r3));
        args
    }
}

/// Decoder for sol_alloc_free_ syscall.
pub struct SolAllocFreeDecoder;

impl SyscallDecoder for SolAllocFreeDecoder {
    fn name(&self) -> &'static str {
        "sol_alloc_free_"
    }

    fn decode_args(
        &self,
        r1: u64,
        r2: u64,
        _r3: u64,
        _r4: u64,
        _r5: u64,
        _memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String> {
        let mut args = HashMap::new();
        args.insert("size".to_string(), format!("{}", r1));
        args.insert("free_ptr".to_string(), format!("{:#x}", r2));
        args
    }
}

/// Decoder for sol_invoke_signed_c syscall (CPI).
pub struct SolInvokeSignedDecoder;

impl SyscallDecoder for SolInvokeSignedDecoder {
    fn name(&self) -> &'static str {
        "sol_invoke_signed_c"
    }

    fn decode_args(
        &self,
        r1: u64,
        r2: u64,
        r3: u64,
        r4: u64,
        r5: u64,
        _memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String> {
        let mut args = HashMap::new();
        args.insert("instruction_ptr".to_string(), format!("{:#x}", r1));
        args.insert("account_infos_ptr".to_string(), format!("{:#x}", r2));
        args.insert("account_infos_len".to_string(), format!("{}", r3));
        args.insert("signers_seeds_ptr".to_string(), format!("{:#x}", r4));
        args.insert("signers_seeds_len".to_string(), format!("{}", r5));
        args
    }
}

/// Decoder for sol_set_return_data syscall.
pub struct SolSetReturnDataDecoder;

impl SyscallDecoder for SolSetReturnDataDecoder {
    fn name(&self) -> &'static str {
        "sol_set_return_data"
    }

    fn decode_args(
        &self,
        r1: u64,
        r2: u64,
        _r3: u64,
        _r4: u64,
        _r5: u64,
        memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String> {
        let mut args = HashMap::new();
        args.insert("data_ptr".to_string(), format!("{:#x}", r1));
        args.insert("data_len".to_string(), format!("{}", r2));

        if let Some(hex) = read_bytes_as_hex(memory_mapping, r1, r2, 64) {
            args.insert("data_hex".to_string(), hex);
        }
        args
    }
}

/// Decoder for sol_get_return_data syscall.
pub struct SolGetReturnDataDecoder;

impl SyscallDecoder for SolGetReturnDataDecoder {
    fn name(&self) -> &'static str {
        "sol_get_return_data"
    }

    fn decode_args(
        &self,
        r1: u64,
        r2: u64,
        r3: u64,
        _r4: u64,
        _r5: u64,
        _memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String> {
        let mut args = HashMap::new();
        args.insert("return_data_ptr".to_string(), format!("{:#x}", r1));
        args.insert("length".to_string(), format!("{}", r2));
        args.insert("program_id_ptr".to_string(), format!("{:#x}", r3));
        args
    }
}

/// Decoder for sol_sha256 syscall.
pub struct SolSha256Decoder;

impl SyscallDecoder for SolSha256Decoder {
    fn name(&self) -> &'static str {
        "sol_sha256"
    }

    fn decode_args(
        &self,
        r1: u64,
        r2: u64,
        r3: u64,
        _r4: u64,
        _r5: u64,
        _memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String> {
        let mut args = HashMap::new();
        args.insert("vals_ptr".to_string(), format!("{:#x}", r1));
        args.insert("vals_len".to_string(), format!("{}", r2));
        args.insert("result_ptr".to_string(), format!("{:#x}", r3));
        args
    }
}

/// Decoder for sol_keccak256 syscall.
pub struct SolKeccak256Decoder;

impl SyscallDecoder for SolKeccak256Decoder {
    fn name(&self) -> &'static str {
        "sol_keccak256"
    }

    fn decode_args(
        &self,
        r1: u64,
        r2: u64,
        r3: u64,
        _r4: u64,
        _r5: u64,
        _memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String> {
        let mut args = HashMap::new();
        args.insert("vals_ptr".to_string(), format!("{:#x}", r1));
        args.insert("vals_len".to_string(), format!("{}", r2));
        args.insert("result_ptr".to_string(), format!("{:#x}", r3));
        args
    }
}

/// Decoder for sol_blake3 syscall.
pub struct SolBlake3Decoder;

impl SyscallDecoder for SolBlake3Decoder {
    fn name(&self) -> &'static str {
        "sol_blake3"
    }

    fn decode_args(
        &self,
        r1: u64,
        r2: u64,
        r3: u64,
        _r4: u64,
        _r5: u64,
        _memory_mapping: &MemoryMapping,
    ) -> HashMap<String, String> {
        let mut args = HashMap::new();
        args.insert("vals_ptr".to_string(), format!("{:#x}", r1));
        args.insert("vals_len".to_string(), format!("{}", r2));
        args.insert("result_ptr".to_string(), format!("{:#x}", r3));
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = SyscallDecoderRegistry::new();
        // Check that built-in decoders are registered
        assert!(registry.decoders.contains_key("sol_log_"));
        assert!(registry.decoders.contains_key("sol_memcpy_"));
        assert!(registry.decoders.contains_key("sol_invoke_signed_c"));
    }

    #[test]
    fn test_fallback_decode() {
        let registry = SyscallDecoderRegistry::new();
        // Unknown syscall should return raw args
        let args = [1u64, 2, 3, 4, 5];
        let decoded = registry.decode(
            "unknown_syscall",
            args,
            &MemoryMapping::Identity,
        );
        assert_eq!(decoded.get("r1"), Some(&"0x1".to_string()));
        assert_eq!(decoded.get("r5"), Some(&"0x5".to_string()));
    }
}
