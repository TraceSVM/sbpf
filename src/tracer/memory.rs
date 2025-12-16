//! Memory region classification utilities for semantic tracing.
//!
//! This module provides functions to classify VM addresses into semantic
//! memory regions (stack, heap, input, etc.) for better trace understanding.

use crate::ebpf::MM_REGION_SIZE;
use super::types::{MemoryRegionType, MAX_MEMORY_VALUE_BYTES};

/// Classify a VM pointer into its memory region.
///
/// The SBPF memory model uses the upper bits of the address to identify
/// the memory region:
/// - Region 0: Rodata (read-only data, SBPFv3)
/// - Region 1: Bytecode (program code + rodata in older versions)
/// - Region 2: Stack (local variables, call frames)
/// - Region 3: Heap (dynamic allocations)
/// - Region 4: Input (account data in Solana)
pub fn classify_pointer(vm_addr: u64) -> MemoryRegionType {
    // Use region index (upper bits of address in 4GB-aligned mapping)
    let region_index = vm_addr / MM_REGION_SIZE;

    match region_index {
        0 => MemoryRegionType::Rodata,    // MM_RODATA_START = 0
        1 => MemoryRegionType::Bytecode,  // MM_BYTECODE_START = 4GB
        2 => MemoryRegionType::Stack,     // MM_STACK_START = 8GB
        3 => MemoryRegionType::Heap,      // MM_HEAP_START = 12GB
        4 => MemoryRegionType::Input,     // MM_INPUT_START = 16GB
        _ => MemoryRegionType::Unknown,
    }
}

/// Format bytes as a hex string, truncating if necessary.
///
/// Returns the hex string and a boolean indicating whether truncation occurred.
pub fn format_hex_value(bytes: &[u8], max_len: usize) -> (String, bool) {
    let truncated = bytes.len() > max_len;
    let slice = if truncated { &bytes[..max_len] } else { bytes };
    (hex::encode(slice), truncated)
}

/// Format bytes as a hex string using the default maximum length.
pub fn format_hex_value_default(bytes: &[u8]) -> (String, bool) {
    format_hex_value(bytes, MAX_MEMORY_VALUE_BYTES)
}

/// Get human-readable region name for a VM address.
pub fn get_region_name(vm_addr: u64) -> &'static str {
    match classify_pointer(vm_addr) {
        MemoryRegionType::Rodata => "rodata",
        MemoryRegionType::Bytecode => "bytecode",
        MemoryRegionType::Stack => "stack",
        MemoryRegionType::Heap => "heap",
        MemoryRegionType::Input => "input",
        MemoryRegionType::Unknown => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ebpf::{MM_BYTECODE_START, MM_HEAP_START, MM_INPUT_START, MM_STACK_START};

    #[test]
    fn test_classify_pointer() {
        // Test each region boundary
        assert_eq!(classify_pointer(0), MemoryRegionType::Rodata);
        assert_eq!(classify_pointer(0x1000), MemoryRegionType::Rodata);

        assert_eq!(classify_pointer(MM_BYTECODE_START), MemoryRegionType::Bytecode);
        assert_eq!(
            classify_pointer(MM_BYTECODE_START + 0x1000),
            MemoryRegionType::Bytecode
        );

        assert_eq!(classify_pointer(MM_STACK_START), MemoryRegionType::Stack);
        assert_eq!(
            classify_pointer(MM_STACK_START + 0x1000),
            MemoryRegionType::Stack
        );

        assert_eq!(classify_pointer(MM_HEAP_START), MemoryRegionType::Heap);
        assert_eq!(
            classify_pointer(MM_HEAP_START + 0x1000),
            MemoryRegionType::Heap
        );

        assert_eq!(classify_pointer(MM_INPUT_START), MemoryRegionType::Input);
        assert_eq!(
            classify_pointer(MM_INPUT_START + 0x1000),
            MemoryRegionType::Input
        );

        // Test unknown region (beyond region 4)
        assert_eq!(
            classify_pointer(MM_REGION_SIZE * 5),
            MemoryRegionType::Unknown
        );
    }

    #[test]
    fn test_format_hex_value() {
        // Test normal case
        let (hex, truncated) = format_hex_value(&[0xde, 0xad, 0xbe, 0xef], 64);
        assert_eq!(hex, "deadbeef");
        assert!(!truncated);

        // Test truncation
        let large = vec![0xab; 100];
        let (hex, truncated) = format_hex_value(&large, 64);
        assert!(truncated);
        assert_eq!(hex.len(), 128); // 64 bytes * 2 hex chars

        // Test empty
        let (hex, truncated) = format_hex_value(&[], 64);
        assert_eq!(hex, "");
        assert!(!truncated);
    }
}
