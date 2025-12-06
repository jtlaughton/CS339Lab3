use rustdb_error::Error;

use crate::disk::disk_manager::DiskManager;
use crate::frame::PageFrame;
use crate::frame_handle::{PageFrameMutHandle, PageFrameRefHandle};
use crate::typedef::{FrameId, PageId};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use bytes::Bytes;
use crate::page::table_page::TablePage;
use crate::Result;

use crate::replacer::replacer::Replacer;

/// Manages page allocation, caching, and eviction in the buffer pool.
#[derive(Debug)]
pub struct BufferPoolManager {
    frames: Vec<PageFrame>, // Storage for all frames in the buffer pool
    page_table: HashMap<PageId, FrameId>, // Maps page IDs to frame IDs
    replacer: Box<dyn Replacer>, // Handles page replacement policy (e.g., LRU)
    free_list: VecDeque<FrameId>, // List of free frames
    disk_manager: Arc<Mutex<DiskManager>>, // Manages reading/writing pages to disk
}

impl BufferPoolManager {
    /// Initializes the buffer pool with a given size.
    pub(crate) fn new(
        pool_size: usize,
        disk_manager: Arc<Mutex<DiskManager>>,
        replacer: Box<dyn Replacer>,
    ) -> Self {
        let mut pages = Vec::with_capacity(pool_size);
        pages.resize_with(pool_size, PageFrame::new);

        Self {
            frames: pages,
            page_table: HashMap::new(),
            replacer,
            free_list: (0..pool_size).collect(),
            disk_manager,
        }
    }

    /// Returns a free frame or evicts a page if necessary.
    fn get_free_frame(&mut self) -> Result<FrameId> {
        if let Some(frame_id) = self.free_list.pop_front() {
            return Ok(frame_id);
        }

        // Evict a page if no free frames are available
        let frame_id = self.replacer.evict().ok_or(Error::BufferPoolError(
            "No evictable frame in buffer pool".to_string(),
        ))?;
        let frame = &mut self.frames[frame_id];
        assert_eq!(
            frame.pin_count(),
            0,
            "If page is evicted from replacer, its pin count must be 0."
        );

        // Write dirty page back to disk before eviction
        if frame.is_dirty() {
            let mut disk = self.disk_manager.lock()?;
            disk.write(frame.page_id(), frame.data())?;
        }

        // Remove old page from the page table
        self.page_table.remove(&frame.page_id());

        // Reset the frame for reuse
        frame.reset();

        Ok(frame_id)
    }

    /// Allocates a new page and loads it into a free frame.
    fn create_page(&mut self) -> Result<&mut PageFrame> {
        let frame_id = self.get_free_frame()?;

        let page_id = {
            let mut disk = self.disk_manager.lock()?;
            disk.allocate_page()?
        };

        let frame = &mut self.frames[frame_id];

        frame.set_page_id(page_id);
        frame.increment_pin_count();

        self.page_table.insert(page_id, frame_id);

        // first recorded access should be pinned
        // recording access on create makes sense but we'll see
        self.replacer.record_access(frame_id);

        Ok(frame)
    }

    /// Fetches a mutable reference to a page, loading it from disk if necessary.
    fn fetch_page_mut(&mut self, page_id: PageId) -> Result<&mut PageFrame> {
        let frame_id = match self.page_table.get(&page_id){
            Some(&val) => val,
            None => {
                let fid = self.get_free_frame()?;

                let data = {
                    let mut disk = self.disk_manager.lock()?;
                    disk.read(page_id)?
                        .ok_or_else(|| Error::InvalidInput(format!("Page {} does not exist", page_id)))?
                };

                let frame = &mut self.frames[fid];
                frame.data_mut().copy_from_slice(data.as_ref());
                frame.set_page_id(page_id);

                self.page_table.insert(page_id, fid);

                fid
            }
        };
        let frame = &mut self.frames[frame_id];

        frame.increment_pin_count();
        self.replacer.pin(frame_id);
        self.replacer.record_access(frame_id);

        Ok(frame)
    }

    /// Fetches an immutable reference to a page.
    fn fetch_page(&mut self, page_id: PageId) -> Result<&PageFrame> {
        Ok(self.fetch_page_mut(page_id)?)
    }

    /// Unpins a page, allowing it to be evicted if necessary.
    pub(crate) fn unpin_page(&mut self, page_id: PageId, is_dirty: bool) {
        match self.page_table.get(&page_id) {
            None => {}
            Some(&val) => {
                let frame = &mut self.frames[val];

                if is_dirty{
                    frame.set_dirty(true);
                }

                frame.decrement_pin_count();

                if frame.pin_count() == 0 {
                    self.replacer.unpin(val);
                }
            }
        }
    }

    /// Deletes a page from the buffer pool and disk.
    pub(crate) fn delete_page(&mut self, page_id: PageId) -> Result<()> {
        let mut matched = false;

        match self.page_table.get(&page_id) {
            None => {}
            Some(&val) => {
                let frame = &mut self.frames[val];
                if frame.pin_count() > 0 {
                    return Err(Error::InvalidInput(format!("Page {} cannot be deleted", page_id)));
                }

                self.replacer.remove(val);
                frame.reset();
                self.free_list.push_back(val);
                matched = true;
            }
        };

        if matched{
            self.page_table.remove(&page_id);
        }

        let mut disk = self.disk_manager.lock()?;
        disk.deallocate_page(page_id)?;

        Ok(())
    }

    /// Flushes a specific page to disk.
    pub(crate) fn flush_page(&mut self, page_id: &PageId) -> Result<()> {
        match self.page_table.get(page_id) {
            None => Ok(()),
            Some(&val) => {
                let frame = &mut self.frames[val];

                if frame.is_dirty() {
                    let mut disk = self.disk_manager.lock()?;
                    disk.write(*page_id, frame.data())?;
                    frame.set_dirty(false);
                }

                Ok(())
            }
        }
    }

    /// Returns the total number of frames in the buffer pool.
    fn capacity(&self) -> usize {
        self.frames.len()
    }

    /// Returns the number of available frames.
    pub(crate) fn free_frame_count(&self) -> usize {
        self.free_list.len() + self.replacer.evictable_count()
    }

    /// Returns the pin count of a page, or `None` if it is not in the buffer pool.
    fn get_pin_count(&self, page_id: PageId) -> Option<u16> {
        let frame_id = self.page_table.get(&page_id)?;

        // Retrieve the frame and get the pin count
        Some(self.frames[*frame_id].pin_count())
    }

    /// Creates a new page and returns a handle for it.
    pub(crate) fn create_page_handle(
        bpm: &Arc<RwLock<BufferPoolManager>>,
    ) -> Result<PageFrameMutHandle> {
        let page_frame = {
            let mut bpm_guard = bpm.write()?;
            // SAFETY:
            // This function needs to return a handle that contains both a reference to a
            // page (created via `create_page()`) and the Arc to the BufferPoolManager.
            // However, `create_page()` returns a reference to a field inside the BufferPoolManager,
            // which is currently borrowed by `bpm_guard`. If we try to call
            // `PageFrameMutHandle::new(&bpm, page_frame)` directly, the borrow checker rejects it
            // because the `page_frame` reference is tied to the lifetime of `bpm_guard`
            // (i.e. the entire BufferPoolManager is considered borrowed).
            //
            // To work around this limitation, we temporarily extract a raw pointer from the locked
            // BufferPoolManager. This allows us to call `create_page()` and obtain a reference to the page
            // without having to keep the full `bpm_guard` active. Since we hold exclusive access via
            // `bpm.write().unwrap()`, we know that the page reference is valid and will not be modified
            // by other threads.
            //
            // In summary, we use `unsafe` here solely to bypass the borrow check that prevents
            // splitting the borrow of the BufferPoolManager into two parts:
            // one for the container (bpm) and one for the page frame extracted from it.
            let bpm_ptr = &mut *bpm_guard as *mut BufferPoolManager;
            unsafe { (*bpm_ptr).create_page()? }
        };

        Ok(PageFrameMutHandle::new(&bpm, page_frame))
    }

    /// Fetches a read-only handle to a page.
    pub(crate) fn fetch_page_handle(
        bpm: &Arc<RwLock<BufferPoolManager>>,
        page_id: PageId,
    ) -> Result<PageFrameRefHandle> {
        let page_frame = {
            let mut bpm_guard = bpm.write()?;
            // SAFETY: see `create_page_handle`
            let bpm_ptr = &mut *bpm_guard as *mut BufferPoolManager;
            unsafe { (*bpm_ptr).fetch_page(page_id)? }
        };

        Ok(PageFrameRefHandle::new(&bpm, page_frame))
    }

    /// Fetches a mutable handle to a page.
    pub(crate) fn fetch_page_mut_handle(
        bpm: &Arc<RwLock<BufferPoolManager>>,
        page_id: PageId,
    ) -> Result<PageFrameMutHandle> {
        let page_frame = {
            let mut bpm_guard = bpm.write()?;
            // SAFETY: see `create_page_handle`
            let bpm_ptr = &mut *bpm_guard as *mut BufferPoolManager;
            unsafe { (*bpm_ptr).fetch_page_mut(page_id)? }
        };

        Ok(PageFrameMutHandle::new(&bpm, page_frame))
    }
}

#[cfg(test)]
mod tests {
    use crate::disk::disk_manager::DiskManager;
    use crate::frame_handle::{PageFrameMutHandle, PageFrameRefHandle};
    use crate::page::PAGE_SIZE;
    use crate::replacer::lru_k_replacer::LrukReplacer;
    use crate::{buffer_pool::BufferPoolManager, typedef::PageId};
    use rand::{rng, Rng};
    use serial_test::serial;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::RwLock;
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread;
    use std::time::Duration;

    // Helper function to create a buffer pool manager with `n` pages.
    fn get_bpm_arc_with_pool_size(pool_size: usize) -> Arc<RwLock<BufferPoolManager>> {
        Arc::new(RwLock::new(get_bpm_with_pool_size(pool_size)))
    }

    fn get_bpm_arc_with_pool_size_and_file_name(
        pool_size: usize,
        file_name: &str,
    ) -> Arc<RwLock<BufferPoolManager>> {
        Arc::new(RwLock::new(get_bpm_with_pool_size_and_file_name(
            pool_size, file_name,
        )))
    }

    fn get_bpm_with_pool_size(pool_size: usize) -> BufferPoolManager {
        let disk_manager = Arc::new(Mutex::new(DiskManager::new("test.db").unwrap()));
        let replacer = Box::new(LrukReplacer::new(5));
        BufferPoolManager::new(pool_size, disk_manager, replacer)
    }

    fn get_bpm_with_pool_size_and_file_name(
        pool_size: usize,
        file_name: &str,
    ) -> BufferPoolManager {
        let disk_manager = Arc::new(Mutex::new(DiskManager::new(file_name).unwrap()));
        let replacer = Box::new(LrukReplacer::new(5));
        BufferPoolManager::new(pool_size, disk_manager, replacer)
    }

    // Helper function to create `n` pages in the buffer pool.
    fn create_n_pages(bpm: &Arc<RwLock<BufferPoolManager>>, n: usize) -> Vec<PageFrameMutHandle> {
        let mut pages = Vec::new();
        for _ in 0..n {
            let page_handle =
                BufferPoolManager::create_page_handle(bpm).expect("Failed to create page");
            pages.push(page_handle);
        }
        pages
    }

    #[test]
    #[serial]
    fn test_bpm_create_pages_beyond_capacity() {
        let pool_size = 10;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        assert_eq!(pool_size, bpm.read().unwrap().free_frame_count());

        {
            let mut handles = vec![];

            // fill the buffer pool with newly created pages
            // these pages should all be pinned
            for i in 0..pool_size {
                let page_handle = BufferPoolManager::create_page_handle(&bpm);
                assert!(page_handle.is_ok());
                handles.push(page_handle);
                assert_eq!(pool_size - i - 1, bpm.read().unwrap().free_frame_count());
            }

            assert_eq!(0, bpm.read().unwrap().free_frame_count());

            {
                // Create a new page when buffer pool has no free frame, should return None
                let page_handle = BufferPoolManager::create_page_handle(&bpm);
                assert!(page_handle.is_err());
            }

            handles.pop();
            assert_eq!(1, bpm.read().unwrap().free_frame_count());

            let page_handle = BufferPoolManager::create_page_handle(&bpm);
            assert!(page_handle.is_ok());
        }
        assert_eq!(pool_size, bpm.read().unwrap().free_frame_count());
    }
    #[test]
    #[serial]
    fn test_bpm_cannot_create_page_beyond_buffer_pool_size() {
        let pool_size = 2;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        // Create and pin two pages.
        let page_handle1 =
            BufferPoolManager::create_page_handle(&bpm).expect("Failed to create page 1");
        let page_handle2 =
            BufferPoolManager::create_page_handle(&bpm).expect("Failed to create page 2");
        let page_id1 = page_handle1.page_id();
        let page_id2 = page_handle2.page_id();

        // Drop the handles to ensure they are unpinned
        drop(page_handle1);
        drop(page_handle2);

        let _read1 =
            BufferPoolManager::fetch_page_handle(&bpm, page_id1).expect("Failed to fetch page 1");
        let _read2 =
            BufferPoolManager::fetch_page_handle(&bpm, page_id2).expect("Failed to fetch page 2");

        // All frames are now pinned, attempt to create another page.
        let result = BufferPoolManager::create_page_handle(&bpm);
        assert!(
            result.is_err(),
            "Should not be able to create a new page when buffer pool is full"
        );
    }

    #[test]
    #[serial]
    fn test_bpm_new_page_evict_frame() {
        let pool_size = 10;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        let mut page_handles = Vec::new(); // Store page handles to prevent dropping

        for _ in 0..pool_size {
            assert!(bpm.read().unwrap().free_frame_count() > 0);
            let page_handle = BufferPoolManager::create_page_handle(&bpm);
            assert!(page_handle.is_ok());
            page_handles.push(page_handle.unwrap()); // Store the handle
        }

        // Free list empty, and no evictable page.
        assert_eq!(bpm.read().unwrap().free_frame_count(), 0);
        assert!(BufferPoolManager::create_page_handle(&bpm).is_err());

        let page_handle = page_handles.pop().unwrap();
        drop(page_handle);
        assert_eq!(bpm.read().unwrap().free_frame_count(), 1);

        let new_page_after_eviction = BufferPoolManager::create_page_handle(&bpm);
        assert!(new_page_after_eviction.is_ok());
        page_handles.push(new_page_after_eviction.unwrap()); // Store the new handle

        assert_eq!(bpm.read().unwrap().free_frame_count(), 0);
        assert!(BufferPoolManager::create_page_handle(&bpm).is_err());
    }

    #[test]
    #[serial]
    fn test_bpm_fetch_page_in_buffer() {
        let pool_size = 10;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        let pages = create_n_pages(&bpm, pool_size);
        let page_ids: Vec<PageId> = pages
            .iter()
            .map(|page_frame_handle| page_frame_handle.page_id())
            .collect();
        drop(pages);
        page_ids.iter().for_each(|&page_id| {
            let page_handle =
                BufferPoolManager::fetch_page_handle(&bpm, page_id).expect("Failed to fetch page");
            assert_eq!(page_handle.page_id(), page_id);
        });
    }

    #[test]
    #[serial]
    fn test_bpm_fetch_page_not_in_buffer() {
        let pool_size = 10;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        // Fill buffer pool to capacity with new pages.
        let page_id_to_evict = {
            let page_handle =
                BufferPoolManager::create_page_handle(&bpm).expect("Failed to create page");
            let page_id = page_handle.page_id();
            page_id
        };

        create_n_pages(&bpm, pool_size - 1);

        // Add another page.
        let _another_page_id = {
            let page_handle =
                BufferPoolManager::create_page_handle(&bpm).expect("Failed to create page");
            let page_id = page_handle.page_id();
            page_id
        };

        // Verify a page was evicted for the new page.
        assert!(!bpm
            .read()
            .unwrap()
            .page_table
            .contains_key(&page_id_to_evict));

        // We should still be able to fetch that evicted page (from disk).
        let fetched_page_handle = BufferPoolManager::fetch_page_handle(&bpm, page_id_to_evict)
            .expect("Failed to fetch page");
        assert_eq!(fetched_page_handle.page_id(), page_id_to_evict);

        // Another fetch of that page (this time from the buffer pool!)
        let fetched_page_handle_again =
            BufferPoolManager::fetch_page_handle(&bpm, page_id_to_evict)
                .expect("Failed to fetch page");
        assert_eq!(fetched_page_handle_again.page_id(), page_id_to_evict);
    }

    #[test]
    #[serial]
    fn test_bpm_unpin_page_changes_dirty_flag() {
        let pool_size = 5;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        let page_id = {
            let mut bpm_write = bpm.write().unwrap();
            let page = bpm_write.create_page().unwrap();
            page.page_id()
        };

        // Initially, the page should not be dirty
        assert!(!bpm.read().unwrap().frames[bpm.read().unwrap().page_table[&page_id]].is_dirty());

        // Unpin the page with `is_dirty = true`
        bpm.write().unwrap().unpin_page(page_id, true);

        // Verify the page is now marked as dirty
        assert!(bpm.read().unwrap().frames[bpm.read().unwrap().page_table[&page_id]].is_dirty());
    }

    #[test]
    #[serial]
    fn test_bpm_unpin_page_not_in_buffer_pool() {
        let bpm = get_bpm_arc_with_pool_size(0);
        let invalid_page_id = 9999;

        // Buffer pool is empty, attempting to unpin should not be allowed
        bpm.write().unwrap().unpin_page(invalid_page_id, false);

        // Since the page does not exist in the buffer pool, there should be no effect
        assert!(!bpm
            .read()
            .unwrap()
            .page_table
            .contains_key(&invalid_page_id));
    }

    #[test]
    #[serial]
    fn test_bpm_unpin_page_decrements_multiple_times() {
        let bpm = get_bpm_arc_with_pool_size(5);

        // Pin count: 1
        let page_id = BufferPoolManager::create_page_handle(&bpm)
            .expect("Failed to create new page")
            .page_id();

        let mut page_handles = Vec::new();
        // Pin count: 25
        for _ in 0..25 {
            let page_handle =
                BufferPoolManager::fetch_page_handle(&bpm, page_id).expect("Failed to fetch page");
            page_handles.push(page_handle);
        }
        assert_eq!(bpm.read().unwrap().get_pin_count(page_id).unwrap(), 25);

        // Pin count: 25 -> 24 -> ... -> 0
        for i in (0..25).rev() {
            let page_handle = page_handles.pop().unwrap();
            drop(page_handle);
            assert_eq!(bpm.read().unwrap().get_pin_count(page_id).unwrap(), i);
        }
    }

    #[test]
    #[serial]
    fn test_bpm_flush_page() {
        let pool_size = 5;
        let bpm = get_bpm_arc_with_pool_size_and_file_name(pool_size, "test.db");

        // Create a new page and modify it
        let page_id = BufferPoolManager::create_page_handle(&bpm)
            .expect("Failed to create new page")
            .page_id();

        let data = b"Test data";
        let page_data = {
            let mut page_handle = BufferPoolManager::fetch_page_mut_handle(&bpm, page_id)
                .expect("Failed to fetch page for writing");
            page_handle.write(0, data);
            page_handle.data().to_vec()
        };

        // Flush the page to disk
        bpm.write()
            .unwrap()
            .flush_page(&page_id)
            .expect("Failed to flush page");

        // Fill the buffer pool with new pages
        let mut pages = create_n_pages(&bpm, pool_size);
        // Buffer pool should be full
        assert_eq!(bpm.read().unwrap().free_frame_count(), 0);
        // Drop one of the page and trigger unpin
        drop(pages.pop().unwrap());
        // Buffer pool should now have one free frame
        assert_eq!(bpm.read().unwrap().free_frame_count(), 1);

        // Ensure the page is still in the buffer pool and is no longer dirty
        let mut binder = bpm.write().unwrap();
        let frame = binder.fetch_page(page_id).expect("Failed to fetch page");
        assert!(!frame.is_dirty(), "Page should not be dirty after flush");
        assert_eq!(frame.data(), page_data, "Page data should persist");
    }

    #[test]
    #[serial]
    fn test_bpm_evict_flush_page() {
        let pool_size = 5;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        // Create a new page and modify it
        let page_id = BufferPoolManager::create_page_handle(&bpm)
            .expect("Failed to create new page")
            .page_id();

        let data = b"Test data";
        let page_data = {
            let mut page_handle = BufferPoolManager::fetch_page_mut_handle(&bpm, page_id)
                .expect("Failed to fetch page for writing");
            page_handle.write(0, data);
            page_handle.data().to_vec()
        };

        // Fill the buffer pool with new pages
        let mut pages = create_n_pages(&bpm, pool_size);
        // Buffer pool should be full
        assert_eq!(bpm.read().unwrap().free_frame_count(), 0);
        // Drop one of the page and trigger unpin
        drop(pages.pop().unwrap());
        // Buffer pool should now have one free frame
        assert_eq!(bpm.read().unwrap().free_frame_count(), 1);

        // Ensure the page is still in the buffer pool and is no longer dirty
        let mut binder = bpm.write().unwrap();
        // Bring the page back into the buffer pool
        let frame = binder.fetch_page(page_id).expect("Failed to fetch page");
        assert!(!frame.is_dirty(), "Page should not be dirty after flush");
        assert_eq!(frame.data(), page_data, "Page data should persist");
    }

    #[test]
    #[serial]
    fn test_bpm_cannot_delete_pinned_page() {
        let mut bpm = get_bpm_with_pool_size(5);

        // Pin count: 1
        let page = bpm.create_page().unwrap();
        let page_id = page.page_id();

        // Deleting a pinned page should
        assert!(bpm.delete_page(page_id).is_err());

        // Pin count: 0
        bpm.unpin_page(page_id, false);

        assert!(bpm.delete_page(page_id).is_ok());
    }

    #[test]
    #[serial]
    fn test_bpm_very_basic_test() {
        let pool_size = 10;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        // Create a new page
        let pid = {
            let page_handle =
                BufferPoolManager::create_page_handle(&bpm).expect("Failed to create page");
            page_handle.page_id()
        };
        let str_data = "Hello, world!".as_bytes();

        // Check WritePageGuard basic functionality
        {
            let mut write_guard = BufferPoolManager::fetch_page_mut_handle(&bpm, pid)
                .expect("Failed to fetch page for writing");

            write_guard.write(0, str_data);
            let stored_data = &write_guard.data()[..str_data.len()];

            assert_eq!(stored_data, str_data);
        }

        // Check ReadPageGuard basic functionality
        {
            let read_guard = BufferPoolManager::fetch_page_handle(&bpm, pid)
                .expect("Failed to fetch page for reading");

            let stored_data = &read_guard.data()[..str_data.len()];
            assert_eq!(stored_data, str_data);
        }

        // Check ReadPageGuard functionality again
        {
            let read_guard = BufferPoolManager::fetch_page_handle(&bpm, pid)
                .expect("Failed to fetch page for reading again");

            let stored_data = &read_guard.data()[..str_data.len()];
            assert_eq!(stored_data, str_data);
        }

        // Delete page
        let result = bpm.write().unwrap().delete_page(pid);
        assert!(result.is_ok(), "Page deletion failed");
    }

    #[test]
    #[serial]
    fn test_bpm_page_pin_easy_test() {
        let pool_size = 2;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        // Create first page
        let pageid0 = BufferPoolManager::create_page_handle(&bpm)
            .expect("Failed to create page 0")
            .page_id();

        // Ensure it's pinned
        assert_eq!(bpm.read().unwrap().get_pin_count(pageid0), Some(0));

        // Create second page
        let pageid1 = BufferPoolManager::create_page_handle(&bpm)
            .expect("Failed to create page 1")
            .page_id();

        // Ensure both pages are pinned
        assert_eq!(bpm.read().unwrap().get_pin_count(pageid0), Some(0));
        assert_eq!(bpm.read().unwrap().get_pin_count(pageid1), Some(0));

        // Write data to pages
        let str0 = b"page0";
        let str1 = b"page1";

        {
            let mut page0_write = BufferPoolManager::fetch_page_mut_handle(&bpm, pageid0)
                .expect("Failed to fetch page 0 for writing");
            page0_write.write(0, str0);

            let mut page1_write = BufferPoolManager::fetch_page_mut_handle(&bpm, pageid1)
                .expect("Failed to fetch page 1 for writing");
            page1_write.write(0, str1);

            // Ensure pin counts are still 1
            assert_eq!(
                bpm.read().unwrap().get_pin_count(pageid0),
                Some(1),
                "Page 0 should still be pinned"
            );
            assert_eq!(
                bpm.read().unwrap().get_pin_count(pageid1),
                Some(1),
                "Page 1 should still be pinned"
            );

            // Drop the page handles, which should unpin them
            drop(page0_write);
            drop(page1_write);
        }

        // Ensure pages are now unpinned
        assert_eq!(
            bpm.read().unwrap().get_pin_count(pageid0),
            Some(0),
            "Page 0 should be unpinned after dropping the handle"
        );
        assert_eq!(
            bpm.read().unwrap().get_pin_count(pageid1),
            Some(0),
            "Page 1 should be unpinned after dropping the handle"
        );
    }


    #[test]
    #[serial]
    fn test_bpm_page_access() {
        let rounds = 50;
        let bpm = get_bpm_arc_with_pool_size(1);

        // Create a new page
        let pid = BufferPoolManager::create_page_handle(&bpm)
            .expect("Failed to create page")
            .page_id();

        let buf = Arc::new(RwLock::new(vec![0u8; PAGE_SIZE]));

        let writer_bpm = Arc::clone(&bpm);
        let writer_thread = thread::spawn(move || {
            for i in 0..rounds {
                thread::sleep(Duration::from_millis(5));

                {
                    // Use a scoped block to drop the write lock as soon as possible
                    let mut page_handle =
                        BufferPoolManager::fetch_page_mut_handle(&writer_bpm, pid)
                            .expect("Failed to fetch page for writing");
                    let data = i.to_string().into_bytes();
                    page_handle.write(0, &data);
                } // Write lock is dropped here
            }
        });

        for _ in 0..rounds {
            thread::sleep(Duration::from_millis(10));

            // Read page while ensuring data consistency
            let page_data = {
                let page_handle = BufferPoolManager::fetch_page_handle(&bpm, pid)
                    .expect("Failed to fetch page for reading");
                let data = page_handle.data().to_vec();
                data // Copy the data and drop the read lock early
            };

            // Store observed data
            {
                let mut buf_guard = buf.write().unwrap();
                buf_guard.copy_from_slice(&page_data[..PAGE_SIZE]);
            }

            thread::sleep(Duration::from_millis(10));

            // Verify that the data remains unchanged during the read lock
            {
                let buf_guard = buf.read().unwrap();
                assert_eq!(
                    buf_guard[..],
                    page_data[..PAGE_SIZE.min(buf_guard.len())] // Ensure bounds safety
                );
            }
        }

        writer_thread.join().expect("Writer thread panicked");
    }

    #[test]
    #[serial]
    fn test_bpm_contention() {
        let rounds = 1000;
        let pool_size = 10;

        // Build your buffer pool manager.
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        // Create a single page for concurrent writes.
        let pid = {
            let page_handle =
                BufferPoolManager::create_page_handle(&bpm).expect("Failed to create page");
            page_handle.page_id()
        };

        // Spawn 4 writer threads, each writing to the same page.
        let mut threads = vec![];
        for _t_id in 1..=4 {
            let local_bpm = Arc::clone(&bpm);
            threads.push(thread::spawn(move || {
                for i in 0..rounds {
                    // Acquire a mutable handle.
                    let mut page_handle = BufferPoolManager::fetch_page_mut_handle(&local_bpm, pid)
                        .expect("Failed to fetch page for writing");

                    // Write the iteration number as bytes.
                    let data = i.to_string().into_bytes();
                    page_handle.write(0, &data);

                    // Dropping `page_handle` releases the page lock and unpins the page.
                }
            }));
        }

        // Wait for all threads to finish.
        for handle in threads {
            handle.join().expect("Writer thread panicked");
        }
    }

    #[test]
    #[serial]
    fn test_bpm_page_pin_hard() {
        // Build your buffer pool manager with a certain size.
        let bpm = get_bpm_arc_with_pool_size(10);

        let mut page_ids = Vec::new();
        let mut contents = Vec::new();

        // Maps of pinned write/read handles (page_id -> handle).
        let mut pages_write: HashMap<PageId, PageFrameMutHandle> = HashMap::new();
        let mut pages_read: HashMap<PageId, PageFrameRefHandle> = HashMap::new();

        let num_pages = 10;

        // 1) Create 10 pages, each pinned for write. Write i, store handle.
        for i in 0..num_pages {
            // New page -> pinned handle
            let pid = {
                let page_handle =
                    BufferPoolManager::create_page_handle(&bpm).expect("Failed to create page");
                page_handle.page_id()
            };

            // Now fetch page in mut mode for writing.
            let mut page = fetch_page_mut_handle_or_none(&bpm, pid)
                .expect("Expected Some(...) since we have free frames for write");

            // Write i
            let data = i.to_string().into_bytes();
            page.write(0, &data);

            // Keep pinned
            pages_write.insert(pid, page);
            page_ids.push(pid);
            contents.push(i.to_string());
        }

        // 2) For each page, pin_count == 1, then drop handles => pin_count == 0
        for (i, &pid) in page_ids.iter().enumerate() {
            assert_eq!(
                bpm.read().unwrap().get_pin_count(pid),
                Some(1),
                "Pin count should be 1 (pinned) for page {}",
                i
            );
            pages_write.remove(&pid); // Dropping pinned handle => unpin
            assert_eq!(
                bpm.read().unwrap().get_pin_count(pid),
                Some(0),
                "Pin count should be 0 after dropping write handle for page {}",
                i
            );
        }

        // 3) Read each page => pin_count from 0 -> 1 => store read handle
        for (i, &pid) in page_ids.iter().enumerate() {
            let page = fetch_page_handle_or_none(&bpm, pid)
                .expect("fetch_page_handle_or_none => Some(...) expected for read");
            let data = page.data();
            let expected = contents[i].as_bytes();
            assert_eq!(
                &data[..expected.len()],
                expected,
                "Page {} content mismatch",
                i
            );
            assert_eq!(
                bpm.read().unwrap().get_pin_count(pid),
                Some(1),
                "After read, pin_count should be 1 for page {}",
                i
            );
            pages_read.insert(pid, page);
        }

        // 4) Attempt to create new pages, but we have 10 pinned read pages => no free frames => fails
        for _ in 0..num_pages {
            let new_pid = match BufferPoolManager::create_page_handle(&bpm) {
                Ok(h) => h.page_id(),
                Err(_) => 999_999, // dummy
            };
            // read that new_pid => should fail => None
            let page_opt = fetch_page_handle_or_none(&bpm, new_pid);
            assert!(
                page_opt.is_none(),
                "No free frames => fetch_page_handle should return None"
            );
        }

        // 5) Another read of each page => pin_count from 1 -> 2
        for (i, &pid) in page_ids.iter().enumerate() {
            let page2 = fetch_page_handle_or_none(&bpm, pid)
                .expect("Should be able to pin the same page again => second read handle");
            let data = page2.data();
            let expected = contents[i].as_bytes();
            assert_eq!(&data[..expected.len()], expected, "Second read mismatch");
            assert_eq!(
                bpm.read().unwrap().get_pin_count(pid),
                Some(2),
                "Pin count => 2 after second read"
            );
            // Drop page2 immediately => pin_count -> 1
        }
        // Implicit drop => scope ends => pin_count => 1 again

        // 6) Check pin_count is back to 1 for each
        for (i, &pid) in page_ids.iter().enumerate() {
            assert_eq!(
                bpm.read().unwrap().get_pin_count(pid),
                Some(1),
                "Pin count => 1 after dropping second read handle for page {}",
                i
            );
        }

        // 7) Drop read handle for page_ids[4]. => pin_count => 0 => it can be evicted
        {
            let pid4 = page_ids[4];
            pages_read.remove(&pid4);
            // Create new page => success => might evict page4
            let new_pid1 = BufferPoolManager::create_page_handle(&bpm)
                .expect("Failed to create new page1")
                .page_id();

            // read new_pid1 => should succeed => pinned
            let new_page1 = fetch_page_handle_or_none(&bpm, new_pid1)
                .expect("We should be able to read newly created page");
            // Now page4 might be evicted => reading it => None
            let page4_opt = fetch_page_handle_or_none(&bpm, pid4);
            assert!(page4_opt.is_none(), "Page4 got evicted => fetch => None");
            drop(new_page1); // Unpin new_pid1
        }

        // 8) Similarly, drop pages [5], [6], [7] => attempt rewriting them => etc.
        {
            let pid5 = page_ids[5];
            let pid6 = page_ids[6];
            let pid7 = page_ids[7];

            // Drop read handles => pin_count => 0
            pages_read.remove(&pid5);
            pages_read.remove(&pid6);
            pages_read.remove(&pid7);

            // Overwrite them with new data
            let updated5 = b"updatedpage5";
            let updated6 = b"updatedpage6";
            let updated7 = b"updatedpage7";

            // Write updates to page5
            {
                let mut page5 = fetch_page_mut_handle_or_none(&bpm, pid5)
                    .expect("Should be able to pin page5 for writing");
                page5.write(0, updated5);
                // Dropping => unpin
            }

            // Write updates to page6
            {
                let mut page6 = fetch_page_mut_handle_or_none(&bpm, pid6)
                    .expect("Should be able to pin page6 for writing");
                page6.write(0, updated6);
            }

            // Write updates to page7
            {
                let mut page7 = fetch_page_mut_handle_or_none(&bpm, pid7)
                    .expect("Should be able to pin page7 for writing");
                page7.write(0, updated7);
            }

            // After dropping them, each pin count => 0
            assert_eq!(bpm.read().unwrap().get_pin_count(pid5), Some(0));
            assert_eq!(bpm.read().unwrap().get_pin_count(pid6), Some(0));
            assert_eq!(bpm.read().unwrap().get_pin_count(pid7), Some(0));

            // Create a new page => may evict page5, for example
            let new_pid2 = {
                let handle = BufferPoolManager::create_page_handle(&bpm)
                    .expect("Failed to create new page2");
                handle.page_id()
            };
            let new_page2_opt = fetch_page_handle_or_none(&bpm, new_pid2);
            assert!(
                new_page2_opt.is_some(),
                "Should read newly created page => possible eviction of old pages"
            );

            // Verify page5 data (reloaded or in memory)
            let page5_opt = fetch_page_handle_or_none(&bpm, pid5);
            assert!(page5_opt.is_some(), "Page5 should be readable again");
            let page5_handle = page5_opt.unwrap();
            assert_eq!(
                &page5_handle.data()[..updated5.len()],
                updated5,
                "page5 => updated content"
            );

            // Verify page7 data
            let page7_opt = fetch_page_handle_or_none(&bpm, pid7);
            assert!(page7_opt.is_some(), "Page7 should be readable");
            let page7_handle = page7_opt.unwrap();
            assert_eq!(
                &page7_handle.data()[..updated7.len()],
                updated7,
                "page7 => updated content"
            );

            let page6_read_opt = fetch_page_handle_or_none(&bpm, pid6);
            assert!(page6_read_opt.is_some());
            drop(page6_read_opt.unwrap());

            // Drop new_page2 => free a frame
            drop(new_page2_opt);

            // Now we can pin page6 for writing again
            let mut page6_write_opt =
                fetch_page_mut_handle_or_none(&bpm, pid6).expect("Should be able to pin page6 now");
            // Confirm we see the old update
            assert_eq!(
                &page6_write_opt.data()[..updated6.len()],
                updated6,
                "page6 => previously updated content"
            );

            // Write another update
            let updated6_12345 = b"12345updatedpage6";
            page6_write_opt.write(0, updated6_12345);

            // Try creating a new page => might fail if everything is pinned
            let new_pid3 = {
                match BufferPoolManager::create_page_handle(&bpm) {
                    Ok(h) => h.page_id(),
                    Err(_) => 999_999,
                }
            };
            let new_page3 = fetch_page_handle_or_none(&bpm, new_pid3);
            assert!(new_page3.is_some());

            // Drop page7, page6 => free frames
            drop(page7_handle);
            drop(page6_write_opt);

            // Now we can create new_pid3 again => should succeed
            let new_pid3b = BufferPoolManager::create_page_handle(&bpm)
                .expect("Should create page after freeing frames")
                .page_id();

            let new_page3b = fetch_page_handle_or_none(&bpm, new_pid3b);
            assert!(
                new_page3b.is_some(),
                "Now we can read new_pid3b => Some(...)"
            );

            // Re-check page6 => updated6_12345
            let page6_opt2 = fetch_page_handle_or_none(&bpm, pid6);
            assert!(page6_opt2.is_some());
            let page6_handle2 = page6_opt2.unwrap();
            assert_eq!(
                &page6_handle2.data()[..updated6_12345.len()],
                updated6_12345,
                "page6 => new update"
            );

            // If page7 was evicted, we drop new_page3b => try again
            drop(new_page3b);
            let page7_opt2 = fetch_page_handle_or_none(&bpm, pid7);
            assert!(page7_opt2.is_some());
            let page7_handle2 = page7_opt2.unwrap();
            assert_eq!(
                &page7_handle2.data()[..updated7.len()],
                updated7,
                "page7 => updated content"
            );
        }
    }

    fn fetch_page_handle_or_none(
        bpm: &Arc<RwLock<BufferPoolManager>>,
        pid: PageId,
    ) -> Option<PageFrameRefHandle> {
        match BufferPoolManager::fetch_page_handle(bpm, pid) {
            Ok(h) => Some(h),
            Err(_) => None,
        }
    }

    fn fetch_page_mut_handle_or_none(
        bpm: &Arc<RwLock<BufferPoolManager>>,
        pid: PageId,
    ) -> Option<PageFrameMutHandle> {
        match BufferPoolManager::fetch_page_mut_handle(bpm, pid) {
            Ok(h) => Some(h),
            Err(_) => None,
        }
    }

    #[test]
    #[serial]
    fn test_bpm_page_pin_hard_with_random_data() {
        let pool_size = 10;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        // 2) Create a new page (pid0) and write random data to it.
        let pid0 = {
            let page_handle =
                BufferPoolManager::create_page_handle(&bpm).expect("Failed to create page 0");
            page_handle.page_id()
        };

        // Pin the page for writing.
        let mut page0_write = BufferPoolManager::fetch_page_mut_handle(&bpm, pid0)
            .expect("Failed to fetch page0 for writing");

        // Generate random data of PAGE_SIZE length.
        let mut rng = rng();
        let mut random_data = vec![0u8; PAGE_SIZE];
        for byte in &mut random_data {
            *byte = rng.random();
        }
        // Optionally terminate the last few bytes
        if PAGE_SIZE >= 2 {
            random_data[PAGE_SIZE - 2] = 0;
            random_data[PAGE_SIZE - 1] = 0;
        }

        // Write random data to the page and verify it immediately.
        page0_write.write(0, &random_data);
        assert_eq!(&page0_write.data()[..], &random_data[..]);

        // Unpin page0 by dropping it.
        drop(page0_write);

        // 3) Fill up the buffer pool with pinned pages.
        let mut pages = Vec::new();
        for _ in 0..pool_size {
            let new_pid = BufferPoolManager::create_page_handle(&bpm)
                .expect("Failed to create new page")
                .page_id();
            let page_handle = BufferPoolManager::fetch_page_mut_handle(&bpm, new_pid)
                .expect("Failed to fetch newly created page for writing");
            pages.push(page_handle);
        }

        // 4) Verify all pinned pages have pin_count == 1.
        for page in &pages {
            let pid = page.page_id();
            assert_eq!(
                bpm.read().unwrap().get_pin_count(pid),
                Some(1),
                "All pages should be pinned with pin_count == 1"
            );
        }

        // 5) Because the buffer pool is full, creating more pages should fail.
        for _ in 0..pool_size {
            assert!(
                BufferPoolManager::create_page_handle(&bpm).is_err(),
                "Should not be able to create new page when buffer pool is full"
            );
        }

        // 6) Drop the first 5 pages => unpin them => pin_count => 0.
        for _ in 0..pool_size / 2 {
            let pid = pages[0].page_id();
            assert_eq!(bpm.read().unwrap().get_pin_count(pid), Some(1));
            drop(pages.remove(0));
            assert_eq!(bpm.read().unwrap().get_pin_count(pid), Some(0));
        }

        // 7) The remaining pages are still pinned => pin_count == 1.
        for page in &pages {
            let pid = page.page_id();
            assert_eq!(
                bpm.read().unwrap().get_pin_count(pid),
                Some(1),
                "Remaining pages are still pinned with pin_count == 1"
            );
        }

        // 8) Create new pages (evicting the unpinned ones).
        for _ in 0..pool_size / 2 {
            let page_handle = BufferPoolManager::create_page_handle(&bpm)
                .expect("Failed to create page after unpinning");
            pages.push(page_handle);
        }

        // 9) Buffer pool is full again => creating more pages should fail.
        for _ in 0..pool_size {
            assert!(
                BufferPoolManager::create_page_handle(&bpm).is_err(),
                "Should fail to create page => buffer pool full"
            );
        }

        // 10) Drop the next 5 pages => unpin them => pin_count => 0
        for _ in 0..pool_size / 2 {
            let pid = pages[0].page_id();
            assert_eq!(bpm.read().unwrap().get_pin_count(pid), Some(1));
            drop(pages.remove(0));
            assert_eq!(bpm.read().unwrap().get_pin_count(pid), Some(0));
        }

        // 11) Fetch the original page (pid0) in read mode => compare random data.
        {
            let page0_read = BufferPoolManager::fetch_page_handle(&bpm, pid0)
                .expect("Failed to fetch original page0 for reading");
            assert_eq!(
                &page0_read.data()[..],
                &random_data[..],
                "Original random data should remain intact"
            );
            // Optionally drop page0_read => unpin
            drop(page0_read);
        }

        // 12) Drop the last 5 pinned pages => unpin them => pin_count => 0
        for _ in 0..pool_size / 2 {
            let pid = pages[0].page_id();
            assert_eq!(bpm.read().unwrap().get_pin_count(pid), Some(1));
            drop(pages.remove(0));
            assert_eq!(bpm.read().unwrap().get_pin_count(pid), Some(0));
        }
    }

    #[test]
    #[serial]
    fn test_bpm_new_page() {
        let pool_size = 10;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        // We'll store pinned write handles here
        let mut pgs: Vec<PageFrameMutHandle> = Vec::new();

        // 2) Fill up the buffer pool with pinned pages
        for i in 0..pool_size {
            // Create a new page
            let page_id = {
                let handle =
                    BufferPoolManager::create_page_handle(&bpm).expect("Failed to create page");
                handle.page_id()
            };

            // Pin the newly created page in write mode
            let mut page_handle = BufferPoolManager::fetch_page_mut_handle(&bpm, page_id)
                .expect("Failed to fetch newly created page for writing");

            // Write data (e.g., the index `i`) for clarity
            let data = i.to_string().into_bytes();
            page_handle.write(0, &data);

            // Store the pinned handle
            pgs.push(page_handle);
        }

        // 3) All pages should be pinned with pin_count == 1
        for handle in &pgs {
            let pid = handle.page_id();
            assert_eq!(
                bpm.read().unwrap().get_pin_count(pid),
                Some(1),
                "All pages should be pinned => pin_count == 1"
            );
        }

        // 4) The buffer pool is full. Creating more pages should fail
        for _ in 0..100 {
            assert!(
                BufferPoolManager::create_page_handle(&bpm).is_err(),
                "No free frames => must fail to create new page"
            );
        }

        // 5) Unpin (drop) the last 5 pages => they become evictable
        for _ in 0..5 {
            pgs.pop().expect("We have enough pages to pop");
        }

        // 6) Create 5 new pages => pinned again
        for _ in 0..5 {
            let page_id = {
                let handle =
                    BufferPoolManager::create_page_handle(&bpm).expect("Failed to create page");
                handle.page_id()
            };
            let mut page_handle = BufferPoolManager::fetch_page_mut_handle(&bpm, page_id)
                .expect("Failed to fetch new page for writing");
            // Optionally write some data
            let data = b"some new data";
            page_handle.write(0, data);

            pgs.push(page_handle);
        }

        // 7) The buffer pool is full again. Creating more pages should fail
        for _ in 0..100 {
            assert!(
                BufferPoolManager::create_page_handle(&bpm).is_err(),
                "Again no free frames => must fail"
            );
        }

        // 8) Unpin (drop) the last 5 pages => they become evictable
        for _ in 0..5 {
            pgs.pop().expect("We have enough pages to pop");
        }

        // 9) Create 5 more pages => pinned
        for _ in 0..5 {
            let page_id = {
                let handle =
                    BufferPoolManager::create_page_handle(&bpm).expect("Failed to create page");
                handle.page_id()
            };
            let mut page_handle = BufferPoolManager::fetch_page_mut_handle(&bpm, page_id)
                .expect("Failed to fetch page for writing");
            page_handle.write(0, b"some fresh data");
            pgs.push(page_handle);
        }

        // 10) The buffer pool is full again => new pages must fail
        for _ in 0..100 {
            assert!(
                BufferPoolManager::create_page_handle(&bpm).is_err(),
                "No free frames => creation fails"
            );
        }
    }

    // If your concurrency logic is correct, the test completes quickly.
    // If there's a lock ordering issue, you may see a deadlock hang.
    #[test]
    #[serial]
    fn test_bpm_deadlock() {
        let pool_size = 10;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        let pid0 = {
            let page_handle =
                BufferPoolManager::create_page_handle(&bpm).expect("Failed to create page 0");
            page_handle.page_id()
        };

        let pid1 = {
            let page_handle =
                BufferPoolManager::create_page_handle(&bpm).expect("Failed to create page 1");
            page_handle.page_id()
        };

        let guard0 = BufferPoolManager::fetch_page_mut_handle(&bpm, pid0)
            .expect("Failed to fetch pid0 for writing in main thread");

        let start = Arc::new(AtomicBool::new(false));
        let start_for_child = Arc::clone(&start);
        let bpm_for_child = Arc::clone(&bpm);

        // Spawn a child thread that tries to also fetch pid0 in write mode
        let child = thread::spawn(move || {
            // Signal the main thread that we started
            start_for_child.store(true, Ordering::Release);

            // Attempt to fetch pid0 in write mode – could deadlock if concurrency logic is wrong
            let _guard0_child = BufferPoolManager::fetch_page_mut_handle(&bpm_for_child, pid0)
                .expect("Child thread: fetch_page_mut_handle on pid0");
            // If we get here, we successfully pinned pid0. We drop it at end of scope => unpin
        });

        // Wait for child thread to pin page 0
        while !start.load(Ordering::Acquire) {
            // spin, or do thread::sleep, whichever
        }

        // Simulate the main thread doing some work while STILL holding pid0
        thread::sleep(Duration::from_millis(1000));

        // Now we pin pid1 while still holding pid0
        let _guard1 = BufferPoolManager::fetch_page_mut_handle(&bpm, pid1)
            .expect("Main thread: fetch_page_mut_handle on pid1");

        // We let go of pid0 now so the child can proceed
        drop(guard0);

        // Join the child thread
        child.join().expect("Child thread panicked");
    }

    #[test]
    #[serial]
    fn test_bpm_evictable() {
        let rounds = 500;
        let num_readers = 8;

        // Only 1 frame in the buffer pool
        let bpm = get_bpm_arc_with_pool_size(1);

        for i in 0..rounds {
            // We'll use 'winner_pid' to occupy the only available frame,
            // and 'loser_pid' that can never be loaded since 'winner_pid' is pinned.
            let winner_pid = {
                let page_handle = BufferPoolManager::create_page_handle(&bpm)
                    .expect("Failed to create winner page");
                page_handle.page_id()
            };

            let loser_pid = {
                let page_handle = BufferPoolManager::create_page_handle(&bpm)
                    .expect("Failed to create loser page");
                page_handle.page_id()
            };

            // Condition variable and boolean to synchronize readers waiting
            // until the main thread has pinned the winner page.
            let signal = Arc::new((Mutex::new(false), Condvar::new()));

            // Spawn multiple reader threads
            let mut readers = Vec::new();
            for _ in 0..num_readers {
                let bpm_clone = Arc::clone(&bpm);
                let signal_clone = Arc::clone(&signal);
                readers.push(thread::spawn(move || {
                    let (lock, cv) = &*signal_clone;

                    // Wait until main thread sets 'signal = true'
                    let mut guard = lock.lock().unwrap();
                    while !*guard {
                        guard = cv.wait(guard).unwrap();
                    }
                    drop(guard);

                    // 1) Read-latch the winner page
                    let _read_guard = BufferPoolManager::fetch_page_handle(&bpm_clone, winner_pid)
                        .expect("Failed to read-latch the winner page");

                    // 2) We expect no free frames, so loading the loser page should fail.
                    assert!(
                        BufferPoolManager::fetch_page_handle(&bpm_clone, loser_pid).is_err(),
                        "Should not be able to read loser page when only 1 frame is pinned"
                    );

                    // read_guard is dropped here => unpin
                }));
            }

            // Main thread pins the winner page (either read or write) and signals readers.
            let (lock, cv) = &*signal;
            let mut guard = lock.lock().unwrap();
            if i % 2 == 0 {
                // Even iteration: read-latch winner page
                let winner_read = BufferPoolManager::fetch_page_handle(&bpm, winner_pid)
                    .expect("Failed to read-latch winner page");

                // Wake up all readers
                *guard = true;
                cv.notify_all();
                drop(guard);

                // Let readers run, then eventually unpin
                drop(winner_read);
            } else {
                // Odd iteration: write-latch winner page
                let winner_write = BufferPoolManager::fetch_page_mut_handle(&bpm, winner_pid)
                    .expect("Failed to write-latch winner page");

                *guard = true;
                cv.notify_all();
                drop(guard);

                // Let readers run, then eventually unpin
                drop(winner_write);
            }

            // Join all reader threads
            for rdr in readers {
                rdr.join().expect("Reader thread panicked");
            }
        }
    }

    #[test]
    #[serial]
    fn test_bpm_concurrent_writer() {
        let full_runs = 5;
        let more_frames = 256;
        let num_threads = 16;
        let runs = more_frames / num_threads; // number of pages each thread will fill/unpin

        // We'll run multiple “full_runs”
        for _run_idx in 0..full_runs {
            // 1) Build a buffer pool manager with `more_frames` capacity
            let bpm = get_bpm_arc_with_pool_size(more_frames);

            let mut threads = Vec::new();

            for _tid in 0..num_threads {
                let bpm_clone = Arc::clone(&bpm);

                // 2) Create a thread that does the following:
                let handle = thread::spawn(move || {
                    let mut page_ids = Vec::new();
                    let mut pages = Vec::new();

                    // a) Fill the buffer pool with `runs` pages
                    for _i in 0..runs {
                        // Emulate `bpm->NewPage()`
                        let page_id = {
                            // Create page handle
                            let page_handle = BufferPoolManager::create_page_handle(&bpm_clone)
                                .expect("Failed to create page");
                            page_handle.page_id()
                        };

                        page_ids.push(page_id);

                        // Now fetch the page for writing
                        let mut page_handle =
                            BufferPoolManager::fetch_page_mut_handle(&bpm_clone, page_id)
                                .expect("Failed to fetch newly created page");

                        // Write the page_id as string
                        let data = page_id.to_string().into_bytes();
                        page_handle.write(0, &data);

                        // Keep pinned
                        pages.push(page_handle);
                    }

                    // b) Unpin in reverse order, verifying pin_count
                    for i in 0..runs {
                        // The last pinned page in `pages` is at the end
                        let page_id = page_ids[runs - i - 1];
                        assert_eq!(
                            bpm_clone.read().unwrap().get_pin_count(page_id),
                            Some(1),
                            "Expected pin count == 1"
                        );
                        drop(
                            pages
                                .pop()
                                .expect("We should have a pinned page to pop/drop"),
                        );
                        assert_eq!(
                            bpm_clone.read().unwrap().get_pin_count(page_id),
                            Some(0),
                            "After dropping pinned page, pin_count should be 0"
                        );
                    }

                    // c) Read them again to verify data
                    for &pid in &page_ids {
                        {
                            let read_guard = BufferPoolManager::fetch_page_handle(&bpm_clone, pid)
                                .expect("Failed to read-latch page");
                            let stored_data = read_guard.data();
                            let expected_str = pid.to_string();
                            assert_eq!(
                                &stored_data[..expected_str.len()],
                                expected_str.as_bytes(),
                                "Page content mismatch"
                            );

                            // pin_count is 1 while `read_guard` is pinned
                            assert_eq!(
                                bpm_clone.read().unwrap().get_pin_count(pid),
                                Some(1),
                                "Pin count should be 1 in read-latch"
                            );
                            // read_guard automatically drops => unpin
                        }

                        // pin_count back to 0
                        assert_eq!(
                            bpm_clone.read().unwrap().get_pin_count(pid),
                            Some(0),
                            "Pin count should be 0 after dropping read handle"
                        );
                    }

                    // d) Delete all pages
                    for &pid in &page_ids {
                        let result = bpm_clone.write().unwrap().delete_page(pid);
                        assert!(result.is_ok(), "Failed to delete page");
                    }
                });

                threads.push(handle);
            }

            // Join all threads
            for handle in threads {
                handle.join().expect("Writer thread panicked");
            }
        }
    }

    #[test]
    #[serial]
    fn test_bpm_writers_no_observation() {
        let bpm = get_bpm_arc_with_pool_size(1);
        let pid = BufferPoolManager::create_page_handle(&bpm)
            .unwrap()
            .page_id();

        let bpm1 = Arc::clone(&bpm);
        let bpm2 = Arc::clone(&bpm);

        let writer1 = thread::spawn(move || {
            let mut page = BufferPoolManager::fetch_page_mut_handle(&bpm1, pid).unwrap();
            page.write(0, b"Writer1");
            thread::sleep(Duration::from_millis(50));
            assert!(std::str::from_utf8(page.data())
                .unwrap()
                .starts_with("Writer1"));
        });

        let writer2 = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            let mut page = BufferPoolManager::fetch_page_mut_handle(&bpm2, pid).unwrap();
            page.write(0, b"Writer2");
        });

        writer1.join().unwrap();
        writer2.join().unwrap();
    }

    #[test]
    #[serial]
    fn test_bpm_concurrent_reader_writer() {
        let full_runs = 1;
        let num_frames = 64; // The buffer pool size
        let num_writers = 8;
        let num_readers = 8;
        let num_pages = 256;
        let data_length = 100; // Number of bytes the threads will read and write to

        for _run_idx in 0..full_runs {
            // 1) Create a new BPM with `num_frames`
            let bpm = get_bpm_arc_with_pool_size(num_frames);

            // 2) Create and initialize `num_pages`
            let mut page_ids = Vec::new();
            for _i in 0..num_pages {
                // Create a new page
                let pid = BufferPoolManager::create_page_handle(&bpm)
                    .expect("Failed to create page")
                    .page_id();

                page_ids.push(pid);
            }

            // 3) Spawn writer threads
            let mut threads = Vec::new();
            for tid in 0..num_writers {
                let bpm_clone = Arc::clone(&bpm);
                let page_ids_clone = page_ids.clone();
                threads.push(thread::spawn(move || {
                    for &pid in &page_ids_clone {
                        // Pin page in write mode
                        let mut write_guard =
                            BufferPoolManager::fetch_page_mut_handle(&bpm_clone, pid).unwrap();
                        for i in 0..data_length {
                            write_guard.data_mut()[i] = tid;
                            thread::sleep(Duration::from_micros(1));
                        }
                    }
                }));
            }

            // 4) Spawn reader threads
            for _tid in 0..num_readers {
                let bpm_clone = Arc::clone(&bpm);
                let page_ids_clone = page_ids.clone();
                threads.push(thread::spawn(move || {
                    for &pid in &page_ids_clone {
                        // Pin page in read mode
                        let read_guard = BufferPoolManager::fetch_page_handle(&bpm_clone, pid)
                            .expect("Failed to fetch page for reading");
                        let page_current_tid = read_guard.data()[10];
                        for i in 0..data_length {
                            assert_eq!(read_guard.data()[i], page_current_tid);
                            thread::sleep(Duration::from_micros(1));
                        }
                    }
                }));
            }

            for handle in threads {
                handle.join().expect("Thread panicked");
            }
        }
    }

}
