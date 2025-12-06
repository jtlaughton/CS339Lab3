use core::fmt;
use std::sync::{
    atomic::{AtomicU16, Ordering},
    RwLock,
};

use crate::{
    page::{INVALID_PAGE_ID, PAGE_SIZE},
    typedef::PageId,
};

/// Represents a page in the buffer pool with metadata and data storage.
pub struct PageFrame {
    page_id: PageId,       // Unique identifier for the page
    is_dirty: bool,        // Tracks whether the page has been modified
    pin_cnt: AtomicU16,    // Pin count indicating active users (now atomic)
    lock: RwLock<()>,      // Read-Write lock for thread safety
    data: [u8; PAGE_SIZE], // Page data storage
}

impl fmt::Debug for PageFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PageFrame")
            .field("page_id", &self.page_id)
            .field("is_dirty", &self.is_dirty)
            .field("pin_cnt", &self.pin_cnt.load(Ordering::SeqCst))
            .finish()
    }
}

impl PageFrame {
    /// Creates a new, uninitialized page.
    pub(crate) fn new() -> Self {
        Self {
            page_id: INVALID_PAGE_ID,
            is_dirty: false,
            pin_cnt: AtomicU16::new(0),
            lock: RwLock::new(()),
            data: [0; PAGE_SIZE],
        }
    }

    /// Returns the page ID.
    pub fn page_id(&self) -> PageId {
        self.page_id
    }

    /// Checks if the page is dirty.
    pub(crate) fn is_dirty(&self) -> bool {
        self.is_dirty
    }

    /// Returns the current pin count.
    pub(crate) fn pin_count(&self) -> u16 {
        self.pin_cnt.load(Ordering::Acquire)
    }

    /// Provides read-only access to page data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Provides mutable access to page data.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Sets the page ID.
    pub(crate) fn set_page_id(&mut self, page_id: PageId) {
        self.page_id = page_id;
    }

    /// Marks the page as dirty or clean.
    pub(crate) fn set_dirty(&mut self, dirty: bool) {
        self.is_dirty = dirty;
    }

    /// Sets the pin count directly (overwrites whatever was there).
    pub(crate) fn set_pin_count(&mut self, pin_cnt: u16) {
        self.pin_cnt.store(pin_cnt, Ordering::Release);
    }

    /// Increments the pin count by 1.
    pub(crate) fn increment_pin_count(&mut self) {
        self.pin_cnt.fetch_add(1, Ordering::SeqCst);
    }

    /// Decrements the pin count by 1, ensuring it never goes below zero.
    pub(crate) fn decrement_pin_count(&mut self) {
        let old = self.pin_cnt.fetch_sub(1, Ordering::SeqCst);
        if old == 0 {
            panic!(
                "Pin count should not be zero when decrementing. Page id: {}",
                self.page_id()
            );
        }
    }

    /// Resets the page to its initial state.
    pub(crate) fn reset(&mut self) {
        self.page_id = INVALID_PAGE_ID;
        self.pin_cnt.store(0, Ordering::Release);
        self.is_dirty = false;
        self.data.fill(0);
    }

    /// Writes data to the page at the given offset.
    pub(crate) fn write(&mut self, offset: usize, data: &[u8]) {
        if offset + data.len() > PAGE_SIZE {
            panic!("Write out of bounds");
        }
        self.data[offset..offset + data.len()].copy_from_slice(data);
    }

    /// Acquires a read lock on the page.
    pub(crate) fn read_lock(&self) -> std::sync::RwLockReadGuard<'_, ()> {
        self.lock.read().unwrap()
    }

    /// Acquires a write lock on the page.
    pub(crate) fn write_lock(&self) -> std::sync::RwLockWriteGuard<'_, ()> {
        self.lock.write().unwrap()
    }
}
