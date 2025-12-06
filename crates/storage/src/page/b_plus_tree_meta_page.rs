use core::fmt;
use std::{
    mem,
    ops::{Deref, DerefMut},
};

use crate::{
    frame::PageFrame,
    frame_handle::{PageFrameMutHandle, PageFrameRefHandle},
    typedef::PageId,
};
use bytemuck::{Pod, Zeroable};

use super::{b_plus_tree_page_header::BPLUS_TREE_PAGE_HEADER_SIZE, INVALID_PAGE_ID};

/// A meta page containing only a root page ID.
/// This is used to track the root node of a B+ tree.
/// Used to prevent potential race condition under concurrent environment.
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Debug)]
pub(crate) struct BPlusTreeMetaPageData {
    root_page_id: PageId,
}

/// A wrapper for a B+ tree root page stored in a page frame.
/// This struct is generic over a page handle (immutable or mutable) and provides
/// methods to access and modify the root page ID.
pub(crate) struct BPlusTreeMetaPage<T> {
    page_frame_handle: T,
}

impl<T: Deref<Target = PageFrame>> BPlusTreeMetaPage<T> {
    /// Returns the root page ID, or `None` if the page ID is invalid.
    pub(crate) fn root_page_id(&self) -> Option<PageId> {
        let data = self.page_frame_handle.data();
        let start = BPLUS_TREE_PAGE_HEADER_SIZE;
        let end = start + mem::size_of::<BPlusTreeMetaPageData>();

        let root_page_data = bytemuck::from_bytes::<BPlusTreeMetaPageData>(&data[start..end]);

        if root_page_data.root_page_id == INVALID_PAGE_ID {
            None
        } else {
            Some(root_page_data.root_page_id)
        }
    }

    /// Returns the PageId of the underlying page.
    pub(crate) fn page_id(&self) -> PageId {
        self.page_frame_handle.page_id()
    }
}

impl<T: DerefMut<Target = PageFrame> + Deref<Target = PageFrame>> BPlusTreeMetaPage<T> {
    /// Sets the root page ID.
    pub(crate) fn set_root_page_id(&mut self, root_page_id: PageId) {
        let data = self.page_frame_handle.data_mut();
        let start = BPLUS_TREE_PAGE_HEADER_SIZE;
        let end = start + mem::size_of::<BPlusTreeMetaPageData>();
        let root_page = bytemuck::from_bytes_mut::<BPlusTreeMetaPageData>(&mut data[start..end]);
        root_page.root_page_id = root_page_id;
    }
}

/// Immutable alias for a B+ tree meta page.
pub type BPlusTreeMetaPageRef<'a> = BPlusTreeMetaPage<PageFrameRefHandle<'a>>;
/// Mutable alias for a B+ tree meta page.
pub type BPlusTreeMetaPageMut<'a> = BPlusTreeMetaPage<PageFrameMutHandle<'a>>;

impl<'a> From<PageFrameRefHandle<'a>> for BPlusTreeMetaPageRef<'a> {
    fn from(page_frame_handle: PageFrameRefHandle<'a>) -> Self {
        Self { page_frame_handle }
    }
}

impl<'a> From<PageFrameMutHandle<'a>> for BPlusTreeMetaPageMut<'a> {
    fn from(page_frame_handle: PageFrameMutHandle<'a>) -> Self {
        Self { page_frame_handle }
    }
}

impl<T: Deref<Target = PageFrame>> fmt::Display for BPlusTreeMetaPage<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<T: Deref<Target = PageFrame>> fmt::Debug for BPlusTreeMetaPage<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("BPlusTreeMetaPage")
            .field("Root id", &self.root_page_id())
            .field("Page Id", &self.page_id())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::{
        buffer_pool::BufferPoolManager, disk::disk_manager::DiskManager,
        replacer::lru_k_replacer::LrukReplacer,
    };
    use std::sync::{Arc, Mutex, RwLock};

    /// Helper function to create a BufferPoolManager.
    fn get_bpm_with_pool_size(pool_size: usize) -> BufferPoolManager {
        let disk_manager = Arc::new(Mutex::new(DiskManager::new("test.db").unwrap()));
        let replacer = Box::new(LrukReplacer::new(5));
        BufferPoolManager::new(pool_size, disk_manager, replacer)
    }

    fn get_bpm_arc_with_pool_size(pool_size: usize) -> Arc<RwLock<BufferPoolManager>> {
        Arc::new(RwLock::new(get_bpm_with_pool_size(pool_size)))
    }

    #[test]
    #[serial]
    fn test_index_b_plus_tree_root_page_set_get() {
        let bpm = get_bpm_arc_with_pool_size(1);

        // Create a new page handle from the buffer pool.
        let frame_handle = BufferPoolManager::create_page_handle(&bpm).unwrap();
        // Wrap the page handle as a mutable B+ tree root page.
        let mut root_page = BPlusTreeMetaPageMut::from(frame_handle);

        // Initially, the root page ID should be None.
        assert_eq!(
            root_page.root_page_id(),
            None,
            "Initial root_page_id should be None"
        );

        // Set the root page ID to a new value.
        root_page.set_root_page_id(42);

        // Verify that the root page ID has been updated.
        assert_eq!(
            root_page.root_page_id(),
            Some(42),
            "root_page_id should be updated to 42"
        );

        // Change the root page ID again.
        root_page.set_root_page_id(99);

        // Verify that the new root page ID is reflected.
        assert_eq!(
            root_page.root_page_id(),
            Some(99),
            "root_page_id should be updated to 99"
        );
    }
}
