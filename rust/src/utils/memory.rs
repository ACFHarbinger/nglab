use serde::{Deserialize, Serialize};
use sysinfo::System;

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryStats {
    pub rss: u64, // Resident Set Size in bytes
    pub vms: u64, // Virtual Memory Size in bytes
}

pub fn get_memory_usage() -> Option<MemoryStats> {
    let mut sys = System::new_all();
    let pid = sysinfo::get_current_pid().ok()?;

    // Refresh only the current process to keep it fast
    sys.refresh_all();

    if let Some(process) = sys.process(pid) {
        Some(MemoryStats {
            rss: process.memory(),
            vms: process.virtual_memory(),
        })
    } else {
        None
    }
}
