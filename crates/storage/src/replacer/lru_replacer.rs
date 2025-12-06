use crate::typedef::FrameId;
use std::collections::HashMap;

use super::replacer::Replacer;

#[derive(Debug)]
struct LruNode {
    frame_id: FrameId,
    is_evictable: bool,
    last_accessed_timestamp: u64,
}

#[derive(Debug)]
pub(crate) struct LruReplacer {
    node_store: HashMap<FrameId, LruNode>,
    evictable_count: usize, // Tracks evictable nodes
    current_timestamp: u64,
}

impl LruReplacer {
    pub(crate) fn new() -> Self {
        LruReplacer {
            node_store: HashMap::new(),
            evictable_count: 0,
            current_timestamp: 0,
        }
    }

    fn current_timestamp(&mut self) -> u64 {
        let old_timestamp = self.current_timestamp;
        self.current_timestamp += 1;
        return old_timestamp;
    }
}

impl Replacer for LruReplacer {
    /// Evicts the least recently used evictable frame.
    fn evict(&mut self) -> Option<FrameId> {
        todo!("Implement eviction")
    }

    /// Marks a frame as not evictable (i.e., pinned).
    fn pin(&mut self, frame_id: FrameId) {
       todo!("Implement pin")
    }

    /// Marks a frame as evictable
    fn unpin(&mut self, frame_id: FrameId) {
        todo!("Implement unpin")
    }

    /// Records an access and updates the timestamp.
    /// If the frame_id is new, create a new node.
    fn record_access(&mut self, frame_id: FrameId) {
        todo!("Implement record_access")
    }

    /// Removes a frame from LRU entirely.
    fn remove(&mut self, frame_id: FrameId) {
        todo!("Implement remove")
    }

    /// Returns the number of evictable frames.
    fn evictable_count(&self) -> usize {
        self.evictable_count
    }
}
