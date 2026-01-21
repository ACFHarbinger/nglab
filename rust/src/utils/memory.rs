use serde::{Deserialize, Serialize};
use sysinfo::System;

/// Structure holding memory usage statistics.
#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Resident Set Size (RSS) in bytes.
    pub rss: u64,
    /// Virtual Memory Size (VMS) in bytes.
    pub vms: u64,
}

/// Retrieves the current process's memory usage statistics.
pub fn get_memory_usage() -> Option<MemoryStats> {
    let mut sys = System::new_all();
    let pid = sysinfo::get_current_pid().ok()?;

    // Refresh only the current process to keep it fast
    sys.refresh_all();

    sys.process(pid).map(|process| MemoryStats {
        rss: process.memory(),
        vms: process.virtual_memory(),
    })
}
