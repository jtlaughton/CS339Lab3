use std::fmt::Debug;

use crate::typedef::FrameId;

pub trait Replacer: Send + Sync + Debug {
    /// Marks a frame as unpinned, making it eligible for eviction.
    fn unpin(&mut self, frame_id: FrameId);

    /// Marks a frame as pinned, preventing it from being evicted.
    fn pin(&mut self, frame_id: FrameId);

    /// Record the event that the given frame id is accessed at current timestamp.
    /// Create a new entry if frame id has not been seen before.
    fn record_access(&mut self, frame_id: FrameId);

    /// Attempts to evict a page in frame based on the replacement policy.
    /// Returns `Some(frame_id)` if a page in frame is evicted, otherwise `None`.
    fn evict(&mut self) -> Option<FrameId>;

    /// Returns the number of evictable frames in the replacer.
    fn evictable_count(&self) -> usize;

    /// Removes a page from the replacer. This should only be called on a page that is evictable
    fn remove(&mut self, frame_id: FrameId);
}
