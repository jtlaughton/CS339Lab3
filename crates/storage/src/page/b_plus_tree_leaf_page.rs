use bytemuck::{Pod, Zeroable};
use core::fmt;
use rustdb_error::Error;
use std::cmp::Ordering;
use std::mem;
use std::ops::{Deref, DerefMut};

use super::b_plus_tree_page_header::{
    BPlusTreePageHeader, BPlusTreePageType, BPLUS_TREE_PAGE_HEADER_SIZE,
};
use super::INVALID_PAGE_ID;
use crate::frame::PageFrame;
use crate::frame_handle::{PageFrameMutHandle, PageFrameRefHandle};
use crate::index::b_plus_tree::{BPlusTreeKey, KeyComparator, KEY_SIZE, MAX_SLOT_CNT};
use crate::typedef::PageId;
use crate::Result;
use rustdb_catalog::schema::RecordId;

/// The on-disk layout of a B+ tree leaf page.
/// Following the header, we have an array of fixed-size keys and an array of RecordIds.
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
pub(crate) struct BPlusTreeLeafPageData {
    header: BPlusTreePageHeader,
    /// The key array.
    key_array: [BPlusTreeKey; MAX_SLOT_CNT],
    /// The record id array. Conventionally there is one record id per slot.
    rid_array: [RecordId; MAX_SLOT_CNT],
    /// The page id of the next leaf page.
    next_page_id: PageId,
    _padding: [u8; 4],
}

pub(crate) const BPLUS_TREE_LEAF_PAGE_DATA_SIZE: usize = mem::size_of::<BPlusTreeLeafPageData>();

/// A wrapper for a B+ tree leaf page stored in a page frame.
pub(crate) struct BPlusTreeLeafPage<T> {
    page_frame_handle: T,
}

impl<T: Deref<Target = PageFrame>> BPlusTreeLeafPage<T> {
    /// Returns the PageId of the underlying page.
    pub(crate) fn page_id(&self) -> PageId {
        self.page_frame_handle.page_id()
    }

    /// Returns the page type.
    pub(crate) fn page_type(&self) -> BPlusTreePageType {
        self.header().page_type()
    }

    /// Returns an immutable reference to the leaf page header.
    pub(crate) fn header(&self) -> &BPlusTreePageHeader {
        let data = self.page_frame_handle.data();
        bytemuck::from_bytes(&data[..BPLUS_TREE_PAGE_HEADER_SIZE])
    }

    /// Returns an immutable reference to the leaf page data.
    fn data(&self) -> &BPlusTreeLeafPageData {
        let data = self.page_frame_handle.data();
        let start = BPLUS_TREE_PAGE_HEADER_SIZE;
        let end = start + BPLUS_TREE_LEAF_PAGE_DATA_SIZE;
        bytemuck::from_bytes(&data[start..end])
    }

    /// Returns the next page id of this leaf page.
    pub(crate) fn next_page_id(&self) -> Option<PageId> {
        if self.data().next_page_id == INVALID_PAGE_ID {
            None
        } else {
            Some(self.data().next_page_id)
        }
    }

    /// Returns an immutable slice representing the key array.
    pub(crate) fn key_array(&self) -> &[BPlusTreeKey] {
        let max_size = self.max_size();
        let key_array = &self.data().key_array;
        let len = key_array.len().min(max_size);
        &key_array[..len]
    }

    /// Returns an immutable slice representing the record id array.
    pub(crate) fn rid_array(&self) -> &[RecordId] {
        let max_size = self.max_size();
        let rid_array = &self.data().rid_array;
        let len = rid_array.len().min(max_size);
        &rid_array[..len]
    }

    /// Returns the current number of slots used.
    /// (Slots are counted including the reserved index 0.)
    pub(crate) fn size(&self) -> usize {
        self.header().size()
    }

    pub(crate) fn max_size(&self) -> usize {
        self.header().max_size()
    }

    pub(crate) fn min_size(&self) -> usize {
        self.max_size() / 2
    }

    /// A helper to find the index at which to insert a key.
    /// This helper returns the index where the new key should be inserted to maintain sorted order.
    /// If the key already exist return `None`.
    fn find_insert_index(
        &self,
        key: &BPlusTreeKey,
        comparator: &dyn KeyComparator,
    ) -> Option<usize> {
        let size = self.size();
        for i in 0..size {
            match comparator.compare(&self.key_array()[i], key) {
                Ordering::Greater => {
                    return Some(i);
                }
                Ordering::Equal => {
                    return None;
                }
                Ordering::Less => {}
            }
        }
        // Otherwise, insert at the end.
        Some(size)
    }

    /// Looks up a given key in the leaf page.
    /// Returns `Some(RecordId)` if a matching key is found, or `None` otherwise.
    pub(crate) fn lookup(
        &self,
        key: &BPlusTreeKey,
        comparator: &dyn KeyComparator,
    ) -> Option<RecordId> {
        Some(self.rid_array()[self.find_key_index(key, comparator)?])
    }

    pub(crate) fn find_key_index(
        &self,
        key: &BPlusTreeKey,
        comparator: &dyn KeyComparator,
    ) -> Option<usize> {
        let size = self.size();
        for i in 0..size {
            if comparator.compare(&self.key_array()[i], key) == Ordering::Equal {
                return Some(i);
            }
        }
        None
    }
}

impl<T: DerefMut<Target = PageFrame> + Deref<Target = PageFrame>> BPlusTreeLeafPage<T> {
    /// Sets the maximum number of values that can be stored in the leaf page.
    /// Also set the page type to leaf page.
    pub(crate) fn init(&mut self, max_size: usize) {
        let header = self.header_mut();
        header.set_size(0);
        header.set_max_size(max_size);
        header.set_page_type(BPlusTreePageType::LeafPage);
        self.set_next_page_id(INVALID_PAGE_ID);
    }

    /// Returns a mutable reference to the leaf page header.
    fn header_mut(&mut self) -> &mut BPlusTreePageHeader {
        let data = self.page_frame_handle.data_mut();
        bytemuck::from_bytes_mut(&mut data[..BPLUS_TREE_PAGE_HEADER_SIZE])
    }

    /// Returns a mutable reference to the leaf page data.
    fn data_mut(&mut self) -> &mut BPlusTreeLeafPageData {
        let data = self.page_frame_handle.data_mut();
        let start = BPLUS_TREE_PAGE_HEADER_SIZE;
        let end = start + BPLUS_TREE_LEAF_PAGE_DATA_SIZE;
        bytemuck::from_bytes_mut(&mut data[start..end])
    }

    /// Returns a mutable slice representing the key array.
    pub(crate) fn key_array_mut(&mut self) -> &mut [BPlusTreeKey] {
        let max_size = self.max_size();
        let key_array = &mut self.data_mut().key_array;
        let len = key_array.len().min(max_size);
        &mut key_array[..len]
    }

    /// Returns a mutable slice representing the record id array.
    pub(crate) fn rid_array_mut(&mut self) -> &mut [RecordId] {
        let max_size = self.max_size();
        let rid_array = &mut self.data_mut().rid_array;
        let len = rid_array.len().min(max_size);
        &mut rid_array[..len]
    }

    /// Sets the number of slots used.
    pub(crate) fn set_size(&mut self, new_size: usize) {
        self.header_mut().set_size(new_size);
    }

    pub(crate) fn set_next_page_id(&mut self, next_page_id: PageId) {
        self.data_mut().next_page_id = next_page_id;
    }

    /// Inserts a key-value pair into the leaf page while maintaining
    /// order, shifting entries as needed.
    /// Returns an error if the key already exists,
    /// otherwise, returns the new size after insertion.
    pub(crate) fn insert(
        &mut self,
        key: &BPlusTreeKey,
        val: RecordId,
        comparator: &dyn KeyComparator,
    ) -> Result<usize> {
        let current_size = self.size(); // Number of slots used (including reserved index 0)
        let insert_idx = self
            .find_insert_index(&key, comparator)
            .ok_or_else(|| Error::InvalidInput(format!("Key already exists {:#?}", key)))?;
        let new_size = current_size + 1;
        self.set_size(new_size);
        // Shift entries to the right to make space for the new key and value.
        for i in (insert_idx + 1..new_size).rev() {
            self.key_array_mut()[i] = self.key_array()[i - 1];
            self.rid_array_mut()[i] = self.rid_array()[i - 1];
        }
        // Insert the new key and record ID at the found index.
        self.key_array_mut()[insert_idx] = *key;
        self.rid_array_mut()[insert_idx] = val;
        Ok(new_size)
    }

    /// Remove the entry at the given index (shifting subsequent keys and record IDs).
    /// Index 0 is reserved and cannot be removed.
    pub(crate) fn remove_at(&mut self, index: usize) -> Result<()> {
        let size = self.size();
        if index >= size || index == 0 {
            return Err(Error::InvalidInput(format!(
                "remove_at: invalid index {} (size = {})",
                index, size
            )));
        }
        for i in index..(size - 1) {
            self.key_array_mut()[i] = self.key_array()[i + 1];
            self.rid_array_mut()[i] = self.rid_array()[i + 1];
        }
        self.key_array_mut()[size - 1] = [0u8; KEY_SIZE];
        self.rid_array_mut()[size - 1] = RecordId::default();
        self.set_size(size - 1);
        Ok(())
    }

    /// Searches for a matching key and, if found, removes the entry.
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

    pub(crate) fn move_half_to(&mut self, other: &mut BPlusTreeLeafPageMut, next_page_id: PageId) {
        other.init(self.max_size());

        let move_size = (self.size() + 1) / 2;
        let start_index = self.size() / 2;
        for i in 0..move_size {
            other.key_array_mut()[i] = self.key_array()[start_index + i];
            other.rid_array_mut()[i] = self.rid_array()[start_index + i];
        }

        other.set_size(other.size() + move_size);
        self.set_size(self.size() - move_size);

        // Update the next page pointers
        self.set_next_page_id(next_page_id);
    }

    /// Removes a key and its associated record ID from the leaf page.
    /// If the key exists, it removes it and shifts elements to maintain order.
    /// Returns the updated size of the page after deletion.
    /// If the key doesn't exist, it returns `None`.
    pub(crate) fn remove_and_delete_record(
        &mut self,
        key: &BPlusTreeKey,
        comparator: &dyn KeyComparator,
    ) -> Option<RecordId> {
        let index = self.find_key_index(key, comparator)?;
        let value = self.rid_array()[index];
        let size = self.size();
        for i in index..(size - 1) {
            self.key_array_mut()[i] = self.key_array()[i + 1];
            self.rid_array_mut()[i] = self.rid_array()[i + 1];
        }

        // Zero out the last slot (for safety)
        self.key_array_mut()[size - 1] = [0u8; KEY_SIZE];
        self.rid_array_mut()[size - 1] = RecordId::default();

        // Decrement the size
        self.set_size(size - 1);
        Some(value)
    }

    /// Moves all key & value pairs from this page to recipient page and update the next page id.
    /// Note: MUST MOVE FROM RIGHT TO LEFT
    pub(crate) fn move_all_to(&mut self, other: &mut BPlusTreeLeafPageMut) {
        let other_size = other.size();
        for i in 0..self.size() {
            other.key_array_mut()[other_size + i] = self.key_array()[i];
            other.rid_array_mut()[other_size + i] = self.rid_array()[i];
        }

        other.set_size(other.size() + self.size());
        other.set_next_page_id(self.next_page_id().unwrap_or(INVALID_PAGE_ID));
    }

    /// Moves the last key-value pair of the current leaf page to the front of the recipient.
    pub(crate) fn move_last_to_front_of(&mut self, recipient: &mut BPlusTreeLeafPageMut) {
        assert!(self.size() > 0, "Cannot move from an empty page");

        let recipient_size = recipient.size();
        let current_size = self.size();

        // Decrement current page size
        self.set_size(current_size - 1);

        // Store last key-value pair
        let mut last_key = self.key_array()[current_size - 1];
        let mut last_rid = self.rid_array()[current_size - 1];

        // Shift recipient's elements to the right to make space
        for i in (0..=recipient_size).rev() {
            std::mem::swap(&mut recipient.key_array_mut()[i], &mut last_key);
            std::mem::swap(&mut recipient.rid_array_mut()[i], &mut last_rid);
        }

        // Increment recipient size
        recipient.set_size(recipient_size + 1);
    }

    /// Moves the first key-value pair of the current leaf page to the end of the recipient.
    pub(crate) fn move_first_to_end_of(&mut self, recipient: &mut BPlusTreeLeafPageMut) {
        assert!(self.size() > 0, "Cannot move from an empty page");

        let recipient_size = recipient.size();
        let current_size = self.size();

        // Move first key and value to the recipient's end
        recipient.key_array_mut()[recipient_size] = self.key_array()[0];
        recipient.rid_array_mut()[recipient_size] = self.rid_array()[0];

        // Increment recipient size
        recipient.set_size(recipient_size + 1);

        // Shift the remaining keys and values left
        for i in 1..current_size {
            self.key_array_mut()[i - 1] = self.key_array()[i];
            self.rid_array_mut()[i - 1] = self.rid_array()[i];
        }

        // Clear the last slot for safety
        self.key_array_mut()[current_size - 1] = [0u8; KEY_SIZE];
        self.rid_array_mut()[current_size - 1] = RecordId::default();

        // Decrement size of current page
        self.set_size(current_size - 1);
    }
}

impl<T: Deref<Target = PageFrame>> fmt::Display for BPlusTreeLeafPage<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<T: Deref<Target = PageFrame>> fmt::Debug for BPlusTreeLeafPage<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let header = self.header();
        let size = header.size();
        let max_size = header.max_size();

        f.debug_struct("BPlusTreeLeafPage")
            .field("Size", &size)
            .field("Max Size", &max_size)
            .field("Page Id", &self.page_id())
            .field("Key array", &format!("{:?}", &self.key_array()[..size]))
            .field("Rid array", &format!("{:?}", &self.rid_array()[..size]))
            .finish()
    }
}

/// Immutable alias for a B+ tree leaf page.
pub type BPlusTreeLeafPageRef<'a> = BPlusTreeLeafPage<PageFrameRefHandle<'a>>;
/// Mutable alias for a B+ tree leaf page.
pub type BPlusTreeLeafPageMut<'a> = BPlusTreeLeafPage<PageFrameMutHandle<'a>>;

impl<'a> From<PageFrameRefHandle<'a>> for BPlusTreeLeafPageRef<'a> {
    fn from(page_frame_handle: PageFrameRefHandle<'a>) -> Self {
        Self { page_frame_handle }
    }
}

impl<'a> From<PageFrameMutHandle<'a>> for BPlusTreeLeafPageMut<'a> {
    fn from(page_frame_handle: PageFrameMutHandle<'a>) -> Self {
        Self { page_frame_handle }
    }
}
