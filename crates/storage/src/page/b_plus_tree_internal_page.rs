use bytemuck::{Pod, Zeroable};
use core::fmt;
use rustdb_error::Error;
use std::cmp::Ordering;
use std::ops::{Deref, DerefMut};
use std::{mem, usize};

use crate::frame::PageFrame;
use crate::frame_handle::{PageFrameMutHandle, PageFrameRefHandle};
use crate::index::b_plus_tree::{BPlusTreeKey, KeyComparator, KEY_SIZE, MAX_SLOT_CNT};
use crate::typedef::PageId;
use crate::Result;

use super::b_plus_tree_page_header::{
    BPlusTreePageHeader, BPlusTreePageType, BPLUS_TREE_PAGE_HEADER_SIZE,
};

/// The on-disk layout of a B+ tree internal page.
/// Following the header, we have an array of fixed-size keys and an array of child pointers (PageIds).
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
pub(crate) struct BPlusTreeInternalPageData {
    header: BPlusTreePageHeader,
    /// key_array[0] is reserved/invalid by convention.
    key_array: [BPlusTreeKey; MAX_SLOT_CNT],
    /// The child pointer array. Conventionally there is one pointer per slot.
    pid_array: [PageId; MAX_SLOT_CNT],
}

pub(crate) const BPLUS_TREE_INTERNAL_PAGE_SIZE: usize = mem::size_of::<BPlusTreeInternalPageData>();

/// A wrapper for a B+ tree internal page stored in a page frame.
/// This struct is generic over a page handle (immutable or mutable) and provides
/// methods to access the internal page header, key array, child pointer array,
/// and operations such as lookup and removal.
pub(crate) struct BPlusTreeInternalPage<T> {
    page_frame_handle: T,
}

impl<T: Deref<Target = PageFrame>> BPlusTreeInternalPage<T> {
    /// Returns the PageId of the underlying page.
    pub(crate) fn page_id(&self) -> PageId {
        self.page_frame_handle.page_id()
    }

    /// Returns an immutable reference to the internal page header.
    pub(crate) fn header(&self) -> &BPlusTreePageHeader {
        let data = self.page_frame_handle.data();
        bytemuck::from_bytes(&data[..BPLUS_TREE_PAGE_HEADER_SIZE])
    }

    pub(crate) fn data(&self) -> &BPlusTreeInternalPageData {
        let data = self.page_frame_handle.data();
        let start = BPLUS_TREE_PAGE_HEADER_SIZE;
        let end = start + BPLUS_TREE_INTERNAL_PAGE_SIZE;
        bytemuck::from_bytes(&data[start..end])
    }

    pub(crate) fn page_type(&self) -> BPlusTreePageType {
        self.header().page_type()
    }

    /// Returns an immutable slice representing the key array.
    pub(crate) fn key_array(&self) -> &[BPlusTreeKey] {
        let max_size = self.max_size();
        let key_array = &self.data().key_array;
        let len = key_array.len().min(max_size);
        &key_array[..len]
    }

    /// Returns an immutable slice representing the child pointer array.
    pub(crate) fn pid_array(&self) -> &[PageId] {
        let max_size = self.max_size();
        let pid_array = &self.data().pid_array;
        let len = pid_array.len().min(max_size);
        &pid_array[..len]
    }

    /// Returns the current number of slots used.
    pub(crate) fn size(&self) -> usize {
        self.header().size()
    }

    /// Returns the maximum number of values that can be stored in this leaf page.
    pub(crate) fn max_size(&self) -> usize {
        self.header().max_size()
    }

    /// Gets the minimum number of values that should be stored.
    pub(crate) fn min_size(&self) -> usize {
        (self.max_size() + 1) / 2
    }

    pub(crate) fn find_pid_index(&self, value: PageId) -> Option<usize> {
        // Find the index where value is stored in the pid_array
        self.pid_array().iter().position(|&v| v == value)
    }

    /// Lookup for a given key.
    ///
    /// Starting at index 1 (since key_array[0] is invalid), it finds the first key greater than the
    /// search key and returns the child pointer from the preceding slot. If none is found, it returns
    /// the rightmost child pointer.
    pub(crate) fn lookup(&self, key: &BPlusTreeKey, comparator: &dyn KeyComparator) -> PageId {
        let size = self.size();
        // Implement binary search if you want.
        for i in 1..size {
            if comparator.compare(&self.key_array()[i], &key) == Ordering::Greater {
                return self.pid_array()[i - 1];
            }
        }
        self.pid_array()[size - 1]
    }
}

impl<T: DerefMut<Target = PageFrame> + Deref<Target = PageFrame>> BPlusTreeInternalPage<T> {
    /// Sets the maximum number of values that can be stored in the leaf page.
    /// Also set the page type to internal page.
    pub(crate) fn init(&mut self, max_size: usize) {
        self.header_mut().set_max_size(max_size);
        self.header_mut()
            .set_page_type(BPlusTreePageType::InternalPage);
    }

    /// Returns a mutable reference to the internal page header.
    pub(crate) fn header_mut(&mut self) -> &mut BPlusTreePageHeader {
        let data = self.page_frame_handle.data_mut();
        bytemuck::from_bytes_mut(&mut data[..BPLUS_TREE_PAGE_HEADER_SIZE])
    }

    pub(crate) fn data_mut(&mut self) -> &mut BPlusTreeInternalPageData {
        let data_mut = self.page_frame_handle.data_mut();
        let start = BPLUS_TREE_PAGE_HEADER_SIZE;
        let end = start + BPLUS_TREE_INTERNAL_PAGE_SIZE;
        bytemuck::from_bytes_mut(&mut data_mut[start..end])
    }

    /// Returns a mutable slice representing the key array.
    pub(crate) fn key_array_mut(&mut self) -> &mut [BPlusTreeKey] {
        let max_size = self.max_size();
        let key_array = &mut self.data_mut().key_array;
        let len = key_array.len().min(max_size);
        &mut key_array[..len]
    }

    /// Returns a mutable slice representing the child pointer array.
    pub(crate) fn pid_array_mut(&mut self) -> &mut [PageId] {
        let max_size = self.max_size();
        let pid_array = &mut self.data_mut().pid_array;
        let len = pid_array.len().min(max_size);
        &mut pid_array[..len]
    }

    /// Sets the size (number of slots used).
    pub(crate) fn set_size(&mut self, new_size: usize) {
        if new_size == 0 {
            panic!("what");
        }
        self.header_mut().set_size(new_size);
    }

    /// Remove the entry at the given index (shifting subsequent keys and child pointers).
    /// Index 0 is reserved and cannot be removed.
    pub(crate) fn remove_at(&mut self, index: usize) -> Result<()> {
        assert!(index != 0);
        let size = self.size();
        // Shift entries left from index to size - 1.
        for i in index..(size - 1) {
            self.key_array_mut()[i] = self.key_array()[i + 1];
            self.pid_array_mut()[i] = self.pid_array()[i + 1];
        }
        // Clear the last slot.
        self.key_array_mut()[size - 1] = [0u8; KEY_SIZE];
        self.pid_array_mut()[size - 1] = 0; // Assuming 0 is an invalid PageId.
        self.set_size(size - 1);
        Ok(())
    }

    /// Searches for a matching key (using the comparator) and, if found, removes the entry.
    pub(crate) fn remove_by_key(
        &mut self,
        key: &BPlusTreeKey,
        comparator: &dyn KeyComparator,
    ) -> Result<()> {
        let size = self.size();
        for i in 1..size {
            if comparator.compare(&self.key_array()[i], key) == Ordering::Equal {
                return self.remove_at(i);
            }
        }
        Err(Error::InvalidInput(format!("Key not found {:?}", key)))
    }

    /// Inserts a new key-value pair right after the pair with value == `old_value`.
    ///
    /// This method shifts elements to the right and places `new_key` and `new_value`
    /// in the correct position. Assumes there is enough space to insert.
    ///
    /// Returns the new size after insertion.
    pub(crate) fn insert_node_after(
        &mut self,
        old_value: PageId,
        new_key: BPlusTreeKey,
        new_value: PageId,
    ) -> usize {
        let current_size = self.size();
        let insert_point = self
            .find_pid_index(old_value)
            .expect("Old value should exist")
            + 1; // Insert right after `old_value`

        let mut new_key_temp = new_key;
        let mut new_value_temp = new_value;
        // Shift keys and values right to make space for the new pair
        for i in insert_point..=current_size {
            std::mem::swap(&mut self.key_array_mut()[i], &mut new_key_temp);
            std::mem::swap(&mut self.pid_array_mut()[i], &mut new_value_temp);
            // self.key_array_mut()[i] = self.key_array()[i - 1];
            // self.pid_array_mut()[i] = self.pid_array()[i - 1];
        }

        // // Insert the new key-value pair at the correct location
        // self.key_array_mut()[insert_point] = new_key;
        // self.pid_array_mut()[insert_point] = new_value;

        // Increase the size of the page
        self.set_size(current_size + 1);

        self.size() // Return the new size
    }

    /// Splits this page at the specified key and moves half of the key-value pairs to the recipient page.
    pub(crate) fn split_to_recipient_page_at(
        &mut self,
        recipient: &mut BPlusTreeInternalPageMut,
        key: &BPlusTreeKey,
        val: PageId,
        comparator: &dyn KeyComparator,
    ) -> BPlusTreeKey {
        let total_size = self.max_size() + 1;
        let mut workspace: Vec<(BPlusTreeKey, PageId)> = Vec::with_capacity(total_size);

        // Copy all keys and values into workspace and insert the new key-value pair in the correct position
        let mut inserted = false;
        for i in 0..self.size() {
            if !inserted && comparator.compare(&key, &self.key_array()[i]) == Ordering::Less {
                workspace.push((*key, val));
                inserted = true;
            }
            workspace.push((self.key_array()[i], self.pid_array()[i]));
        }
        if !inserted {
            workspace.push((*key, val));
        }

        // Split the workspace into two halves
        let split_idx = total_size / 2;
        let middle_key = workspace[split_idx].0;

        // Copy back the first half into the current page
        for i in 0..split_idx {
            self.key_array_mut()[i] = workspace[i].0;
            self.pid_array_mut()[i] = workspace[i].1;
        }
        self.set_size(split_idx);

        // for i in split_idx..self.max_size() {
        //     self.key_array_mut()[i] = [0u8; KEY_SIZE];
        //     self.pid_array_mut()[i] = INVALID_PAGE_ID;
        // }

        // Move the second half to the recipient page
        for i in split_idx..total_size {
            recipient.key_array_mut()[i - split_idx] = workspace[i].0;
            recipient.pid_array_mut()[i - split_idx] = workspace[i].1;
        }
        recipient.set_size(total_size - split_idx);
        recipient.key_array_mut()[0] = [0; 4];

        middle_key
    }

    /// Moves all of key & value pairs from this page to recipient page and update the next page id. MUST MOVE FROM RIGHT TO LEFT
    pub(crate) fn move_all_to(&mut self, other: &mut BPlusTreeInternalPageMut, key: &BPlusTreeKey) {
        self.key_array_mut()[0] = *key;
        for i in 0..self.size() {
            let other_size = other.size();
            other.key_array_mut()[other_size] = self.key_array()[i];
            other.pid_array_mut()[other_size] = self.pid_array()[i];
            other.set_size(other.size() + 1);
        }
    }

    /// Moves the last key-value pair of the current leaf page to the front of the recipient.
    pub(crate) fn move_last_to_front_of(
        &mut self,
        recipient: &mut BPlusTreeInternalPageMut,
        middle_key: &BPlusTreeKey,
    ) {
        assert!(self.size() > 0);

        recipient.key_array_mut()[0] = *middle_key;
        let mut prev_key = recipient.key_array()[0];
        let mut prev_pid = recipient.pid_array()[0];
        let last_key = self.key_array()[self.size() - 1];
        let last_pid = self.pid_array()[self.size() - 1];
        recipient.key_array_mut()[0] = last_key;
        recipient.pid_array_mut()[0] = last_pid;
        // Increment recipient size
        recipient.set_size(recipient.size() + 1);

        for i in 1..=recipient.size() {
            std::mem::swap(&mut recipient.key_array_mut()[i], &mut prev_key);
            std::mem::swap(&mut recipient.pid_array_mut()[i], &mut prev_pid);
        }
        self.set_size(self.size() - 1);
    }

    /// Moves the first key-value pair of the current leaf page to the end of the recipient.
    pub(crate) fn move_first_to_end_of(
        &mut self,
        recipient: &mut BPlusTreeInternalPageMut,
        middle_key: &BPlusTreeKey,
    ) {
        assert!(self.size() > 0);
        let recipient_size = recipient.size();
        recipient.key_array_mut()[recipient_size] = *middle_key;
        recipient.pid_array_mut()[recipient_size] = self.pid_array()[0];
        recipient.set_size(recipient_size + 1);
        for i in 1..self.size() {
            self.key_array_mut()[i - 1] = self.key_array()[i];
            self.pid_array_mut()[i - 1] = self.pid_array()[i];
        }
        self.set_size(self.size() - 1);
    }
}

/// Immutable alias for a B+ tree internal page.
pub(crate) type BPlusTreeInternalPageRef<'a> = BPlusTreeInternalPage<PageFrameRefHandle<'a>>;
/// Mutable alias for a B+ tree internal page.
pub(crate) type BPlusTreeInternalPageMut<'a> = BPlusTreeInternalPage<PageFrameMutHandle<'a>>;

impl<'a> From<PageFrameRefHandle<'a>> for BPlusTreeInternalPageRef<'a> {
    fn from(page_frame_handle: PageFrameRefHandle<'a>) -> Self {
        Self { page_frame_handle }
    }
}

impl<'a> From<PageFrameMutHandle<'a>> for BPlusTreeInternalPageMut<'a> {
    fn from(page_frame_handle: PageFrameMutHandle<'a>) -> Self {
        Self { page_frame_handle }
    }
}

impl<T: Deref<Target = PageFrame>> fmt::Display for BPlusTreeInternalPage<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<T: Deref<Target = PageFrame>> fmt::Debug for BPlusTreeInternalPage<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let header = self.header();
        let size = header.size();
        let max_size = header.max_size();

        f.debug_struct("BPlusTreeInternalPage")
            .field("Size", &size)
            .field("Max Size", &max_size)
            .field("Page Id", &self.page_id())
            .field("Key array", &format!("{:?}", &self.key_array()))
            .field("Pid array", &format!("{:?}", &self.pid_array()))
            .finish()
    }
}

#[cfg(test)]
mod tests {

    use serial_test::serial;

    use super::BPlusTreeInternalPageMut;
    use std::sync::{Arc, Mutex, RwLock};

    use crate::{
        buffer_pool::BufferPoolManager,
        disk::disk_manager::DiskManager,
        index::b_plus_tree::{BPlusTreeKey, LexicographicKeyComparator, KEY_SIZE},
        replacer::lru_k_replacer::LrukReplacer,
    };

    // Create a BufferPoolManager with the given pool size.
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
    fn test_index_b_plus_tree_internal_page_lookup_and_remove() {
        let bpm = get_bpm_arc_with_pool_size(10);

        // Create a new page handle from the buffer pool.
        let frame_handle = BufferPoolManager::create_page_handle(&bpm).unwrap();
        // Wrap the page handle as a mutable B+ tree internal page.
        let mut internal_page = BPlusTreeInternalPageMut::from(frame_handle);

        // Initialize the header.
        // For this test, we set max_size to our constant INTERNAL_PAGE_SLOT_CNT and root_page_id to 1.
        {
            let header = internal_page.header_mut();
            header.set_max_size(4);
        }
        // We plan to have three valid entries. Since index 0 is reserved, we set size to 4.
        internal_page.set_size(4);

        // Prepare some test keys (each is a BPlusTreeKey) and child pointers.
        // By convention, key_array[0] remains unused.
        let key1: BPlusTreeKey = [10; KEY_SIZE]; // all bytes = 10
        let key2: BPlusTreeKey = [20; KEY_SIZE];
        let key3: BPlusTreeKey = [30; KEY_SIZE];

        // For child pointers, we use arbitrary PageId values.
        {
            let key_array_mut = internal_page.key_array_mut();

            // Set key_array[0] as invalid.
            key_array_mut[0] = [0u8; KEY_SIZE];
            key_array_mut[1] = key1;
            key_array_mut[2] = key2;
            key_array_mut[3] = key3;

            let pid_array_mut = internal_page.pid_array_mut();
            pid_array_mut[0] = 100;
            pid_array_mut[1] = 200;
            pid_array_mut[2] = 300;
            pid_array_mut[3] = 400;
        }

        let comparator = LexicographicKeyComparator;

        // --- Test Lookup ---
        // Case 1: Search for a key smaller than key1.
        let search_key: BPlusTreeKey = [5; KEY_SIZE]; // 5 < 10
                                                      // Since no key at index >=1 is > search_key, lookup returns pid_array[0] = 100.
        assert_eq!(internal_page.lookup(&search_key, &comparator), 100);

        // Case 2: Search for a key between key1 and key2.
        let search_key2: BPlusTreeKey = [15; KEY_SIZE]; // between 10 and 20.
                                                        // Loop: i=1: compare(key1, search_key2) = -1; i=2: compare(key2, search_key2) = 1.
                                                        // So lookup returns pid_array[1] = 200.
        assert_eq!(internal_page.lookup(&search_key2, &comparator), 200);

        // Case 3: Search for a key greater than key3.
        let search_key3: BPlusTreeKey = [40; KEY_SIZE];
        // No key > search_key3 found, so lookup returns the rightmost child: pid_array[3] = 400.
        assert_eq!(internal_page.lookup(&search_key3, &comparator), 400);

        // --- Test Removal ---
        // Remove key2 ([20; KEY_SIZE]) from the internal page.
        internal_page.remove_by_key(&key2, &comparator).unwrap();
        // After removal, size should decrease by 1.
        assert_eq!(internal_page.size(), 3);

        // Now, the valid entries should be:
        //   index 1: key1, index 2: key3.
        let keys = internal_page.key_array();
        assert_eq!(keys[1], key1);
        assert_eq!(keys[2], key3);

        // Also, the child pointer array should have shifted.
        let pids = internal_page.pid_array();
        // Expected: index 0 remains 100, index 1 remains 200, and index 2 becomes 400 (key2 and its child pointer 300 were removed).
        assert_eq!(pids[0], 100);
        assert_eq!(pids[1], 200);
        assert_eq!(pids[2], 400);

        // A lookup for a key between key1 and key3 should now return pid_array[1] (child 200).
        let search_key4: BPlusTreeKey = [15; KEY_SIZE];
        assert_eq!(internal_page.lookup(&search_key4, &comparator), 200);
    }
}
