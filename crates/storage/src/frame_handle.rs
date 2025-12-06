use crate::buffer_pool::BufferPoolManager;
use crate::frame::PageFrame;
use core::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// A handle for a read-only `PageFrame`.
///
/// This struct ensures that when the handle is dropped, it automatically unpins
/// the page, allowing it to be evicted if necessary.
pub struct PageFrameRefHandle<'a> {
    bpm: &'a Arc<RwLock<BufferPoolManager>>,
    page_frame: &'a PageFrame,
    lock_guard: RwLockReadGuard<'a, ()>,
}

impl fmt::Debug for PageFrameRefHandle<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PageFrameRefHandle")
            .field("page_frame", &self.page_frame)
            .finish()
    }
}

impl<'a> PageFrameRefHandle<'a> {
    // Creates a new read-only page handle.
    pub(crate) fn new(bpm: &'a Arc<RwLock<BufferPoolManager>>, page_frame: &'a PageFrame) -> Self {
        let fp_ptr = &*page_frame as *const PageFrame;
        // SAFETY:
        // Obtains a read lock on the `PageFrame` using an **unsafe** block.
        let lock_guard = unsafe { (*fp_ptr).read_lock() };
        PageFrameRefHandle {
            bpm,
            page_frame,
            lock_guard,
        }
    }
}

impl<'a> Drop for PageFrameRefHandle<'a> {
    fn drop(&mut self) {
        self.bpm
            .write()
            .unwrap()
            .unpin_page(self.page_frame.page_id(), false);
    }
}

/// Mutable page handle for write access.
pub struct PageFrameMutHandle<'a> {
    bpm: &'a Arc<RwLock<BufferPoolManager>>,
    page_frame: &'a mut PageFrame,
    lock_guard: RwLockWriteGuard<'a, ()>,
}

impl fmt::Debug for PageFrameMutHandle<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PageFrameMutHandle")
            .field("page_frame", &self.page_frame)
            .finish()
    }
}

impl<'a> PageFrameMutHandle<'a> {
    pub(crate) fn new(
        bpm: &'a Arc<RwLock<BufferPoolManager>>,
        page_frame: &'a mut PageFrame,
    ) -> Self {
        let fp_ptr = &mut *page_frame as *mut PageFrame;
        // SAFETY:
        // Obtains a read lock on the `PageFrame` using an **unsafe** block.
        let lock_guard = unsafe { (*fp_ptr).write_lock() };
        PageFrameMutHandle {
            bpm,
            page_frame,
            lock_guard,
        }
    }
}

impl<'a> Drop for PageFrameMutHandle<'a> {
    fn drop(&mut self) {
        self.bpm
            .write()
            .unwrap()
            .unpin_page(self.page_frame.page_id(), true);
    }
}

/// Implement `Deref` for `PageFrameRefHandle` to provide transparent access to `PageFrame`.
impl<'a> Deref for PageFrameRefHandle<'a> {
    type Target = PageFrame;

    fn deref(&self) -> &Self::Target {
        self.page_frame
    }
}

/// Implement `Deref` for `PageFrameMutHandle` to provide transparent access to `PageFrame`.
impl<'a> Deref for PageFrameMutHandle<'a> {
    type Target = PageFrame;

    fn deref(&self) -> &Self::Target {
        self.page_frame
    }
}

/// Implement `DerefMut` for `PageFrameMutHandle` to allow mutable access to `PageFrame`.
impl<'a> DerefMut for PageFrameMutHandle<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.page_frame
    }
}
