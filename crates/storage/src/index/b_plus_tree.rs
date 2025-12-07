use rustdb_catalog::schema::RecordId;
use rustdb_error::Result;
use std::{
    cmp::Ordering, collections::VecDeque, os::unix::process::parent_id, sync::{Arc, RwLock}
};

use crate::frame_handle::PageFrameMutHandle;
use crate::{
    buffer_pool::BufferPoolManager,
    page::{
        b_plus_tree_internal_page::{BPlusTreeInternalPageMut, BPlusTreeInternalPageRef},
        b_plus_tree_leaf_page::{BPlusTreeLeafPageMut, BPlusTreeLeafPageRef},
        b_plus_tree_meta_page::{BPlusTreeMetaPageMut, BPlusTreeMetaPageRef},
        b_plus_tree_page_header::{BPlusTreePageHeader, BPlusTreePageType},
        INVALID_PAGE_ID,
    },
    typedef::PageId,
};

/// Defines the external API for interacting with a B+ Tree index.
///
/// A B+ Tree stores sorted key-value pairs, where:
/// - The key (`BPlusTreeKey`) is a fixed-size byte array used for indexing and lookups.
/// - The value (`RecordId`) identifies the location of the corresponding tuple record.
///
/// These methods define the contract that any B+ Tree implementation must follow
/// to ensure correct indexing behavior.
pub trait BPlusTreeIndex {
    /// Retrieves the value (RecordId) associated with the given key, if it exists.
    /// Returns `None` if the key is not present in the tree.
    fn get(&self, key: &BPlusTreeKey) -> Option<RecordId>;

    /// Inserts a key-value pair into the tree.
    /// The tree should maintain its balanced structure, splitting pages if necessary.
    /// Returns `Ok(())` on success, or `Err` if an error occurs.
    /// If the key already exists in the leaf page, this method returns an error.
    fn insert(&self, key: &BPlusTreeKey, value: RecordId) -> Result<()>;

    /// Deletes the specified key from the tree.
    /// Returns the removed value if the key existed, or `None` if it did not.
    /// The tree must rebalance if pages become too empty after removal.
    fn remove(&self, key: &BPlusTreeKey) -> Option<RecordId>;

    /// Returns an iterator over key-value pairs in sorted order.
    ///
    /// - If `start` is provided, iteration begins from the smallest key greater than or equal to `start`.
    /// - If `start` is `None`, iteration begins from the smallest key in the tree.
    ///
    /// - If `end` is provided, iteration stops before the first key greater than `end`.
    /// - If `end` is `None`, iteration continues to the end of the tree.
    ///
    /// Example usage:
    /// - `range(None, None)` returns all entries in order.
    /// - `range(Some(&start_key), None)` returns all entries from `start_key` onward.
    /// - `range(Some(&start_key), Some(&end_key))` returns entries in the range [`start_key`, `end_key`].
    fn range(
        &self,
        start: Option<&BPlusTreeKey>,
        end: Option<&BPlusTreeKey>,
    ) -> BPlusTreeRangeIterator;
}

/// Fixed key length (for example, 4 bytes).
pub(crate) const KEY_SIZE: usize = 4;
/// Fixed key type for the B+ Tree.
pub(crate) type BPlusTreeKey = [u8; KEY_SIZE];
/// Number of slots in the internal and leaf page.
pub(crate) const MAX_SLOT_CNT: usize = 128;

/// The B+ Tree index implementation.
pub struct BPlusTreeIndexImpl {
    /// The name of the index.
    pub(crate) index_name: String,

    /// The buffer pool manager used to manage pages of the B+ Tree.
    pub(crate) bpm: Arc<RwLock<BufferPoolManager>>,

    /// The page id of the B+ Tree's metadata page.
    /// See `b_plus_tree_meta_page.rs`.
    pub(crate) meta_page_id: PageId,

    /// The maximum number of entries in a leaf page.
    /// See `b_plus_tree_leaf_page.rs`.
    pub(crate) leaf_max_size: usize,

    /// The maximum number of entries in an internal page.
    /// See `b_plus_tree_internal_page.rs`.
    pub(crate) internal_max_size: usize,

    /// The key comparator used for ordering keys.
    pub(crate) comparator: Box<dyn KeyComparator>,
}

/// A comparator for fixed-size keys.
/// `compare` returns Ordering::Less if left < right,
/// Ordering::Equal if equal, and Ordering::Greater if left > right.
pub(crate) trait KeyComparator: Send + Sync {
    fn compare(&self, left: &BPlusTreeKey, right: &BPlusTreeKey) -> Ordering;
}

/// A simple key comparator that compares two fixed-size keys (each [u8; KEY_SIZE]) lexicographically.
pub(crate) struct LexicographicKeyComparator;
impl KeyComparator for LexicographicKeyComparator {
    fn compare(&self, left: &BPlusTreeKey, right: &BPlusTreeKey) -> Ordering {
        left.cmp(right)
    }
}

/// Represents a read-only reference to any type of B+ Tree page (internal, leaf, or meta).
/// Used when traversing the tree without making modifications.
#[derive(Debug)]
enum BPlusTreePageRef<'a> {
    /// Reference to an internal page (non-leaf).
    Internal(BPlusTreeInternalPageRef<'a>),
    /// Reference to a leaf page.
    Leaf(BPlusTreeLeafPageRef<'a>),
    /// Reference to the meta page that stores the root page id.
    Root(BPlusTreeMetaPageRef<'a>),
}

/// Represents a mutable reference to any type of B+ Tree page (internal, leaf, or meta).
/// Used when modifying the tree structure or contents.
#[derive(Debug)]
enum BPlusTreePageMut<'a> {
    /// Mutable reference to an internal page (non-leaf).
    Internal(BPlusTreeInternalPageMut<'a>),
    /// Mutable reference to a leaf page.
    Leaf(BPlusTreeLeafPageMut<'a>),
    /// Mutable reference to the meta page that stores the root page id.
    Root(BPlusTreeMetaPageMut<'a>),
}

/// Describes the purpose of finding a leaf page, either for insertion or deletion.
/// Used to apply the correct safety checks while traversing the tree.
#[derive(Debug, Copy, Clone)]
pub enum FindLeafOption {
    /// The operation is inserting a new key-value pair.
    Insert,
    /// The operation is deleting an existing key-value pair.
    Delete,
    /// The operation is searching for a key-value pair.
    Lookup,
}

/// Tracks the state of a B+ Tree traversal or modification operation.
/// Used to manage the meta page, root page id, and the read/write page sets.s
///
/// Pages stored in `meta_page` and `write_set` hold internal locks
/// (such as read or write locks) through their page handles.
/// These locks remain active as long as the handles are stored in the context.
#[derive(Debug)]
struct BPlusTreeContext<'a> {
    /// Optionally holds a mutable reference to the meta page.
    /// This allows updating the root page id if the root changes (since the meta page points to it).
    meta_page: Option<BPlusTreeMetaPageMut<'a>>,

    /// Cached root page id for the current operation.
    root_page_id: Option<PageId>,

    /// Stack of pages visited in read-only mode (not modified).
    read_set: VecDeque<BPlusTreePageRef<'a>>,

    /// Stack of pages visited in write mode (modified or about to be modified).
    write_set: VecDeque<BPlusTreePageMut<'a>>,
}

/// Implementation of the BPlusTreeContext struct.
impl<'a> BPlusTreeContext<'a> {
    fn new() -> Self {
        Self {
            meta_page: Option::None,
            root_page_id: Option::None,
            read_set: VecDeque::new(),
            write_set: VecDeque::new(),
        }
    }

    /// Releases all held page locks in the current context.
    /// This allows the tree to release resources once it reaches a safe state,
    /// such as when the subtree being modified no longer requires exclusive access.
    fn unlock_all(&mut self) {
        if let Some(meta_page) = self.meta_page.take() {
            drop(meta_page);
        }
        self.write_set.clear();
    }

    // For insert, check if there is enough space in the `next` page.
    // For delete, if it is a non-root page confirm that deleting a record will not cause it to become no longer half-full
    fn is_safe(next: &BPlusTreePageHeader, is_root: bool, option: FindLeafOption) -> bool {
        match option {
            FindLeafOption::Insert => {
                if next.page_type() == BPlusTreePageType::LeafPage {
                    next.size() < next.max_size() - 1
                } else {
                    next.size() < next.max_size()
                }
            }
            FindLeafOption::Delete => {
                // 1. Root needs at least 3 nodes; otherwise, if one is removed, the single child becomes the new root.
                // 2. Non-root nodes just need to be larger than the min size.
                (is_root && next.size() > 2) || next.size() > next.min_size()
            }
            FindLeafOption::Lookup => true,
        }
    }
}

impl BPlusTreeIndexImpl {
    /// Creates and returns a new B+ Tree Index instance given the input parameters.
    ///
    /// We need to instantiate the index's corresponding metadata page (with type
    /// `BPlusTreeMetaPageMut`), which requires dispatching to the buffer pool manager
    /// to create a page handle in which we'll store the metadata page instance.
    ///
    /// * `index_name` - A string that names the index.
    /// * `bpm` - A shared reference to the buffer pool manager.
    /// * `internal_max_size` - Maximum number of keys for internal pages.
    /// * `leaf_max_size` - Maximum number of keys for leaf pages.
    pub fn new(
        index_name: String,
        bpm: Arc<RwLock<BufferPoolManager>>,
        internal_max_size: usize,
        leaf_max_size: usize,
    ) -> Self {
        let meta_page_id = {
            let page_handle = BufferPoolManager::create_page_handle(&bpm).unwrap();
            let meta_page = BPlusTreeMetaPageMut::from(page_handle);
            meta_page.page_id()
        };
        Self {
            index_name,
            bpm,
            meta_page_id,
            internal_max_size,
            leaf_max_size,
            comparator: Box::new(LexicographicKeyComparator),
        }
    }

    /// Initializes a new B+ Tree with a single leaf page containing the input key/value
    /// pair, resetting the given metadata page to correspond with our newly initialized
    /// tree.
    ///
    /// # Arguments
    /// * `meta_page` - A mutable reference to the meta page to update the root pointer.
    /// * `key` - The key to insert as the first entry.
    /// * `value` - The value to associate with the key.
    fn start_new_tree(
        &self,
        meta_page: &mut BPlusTreeMetaPageMut,
        key: &BPlusTreeKey,
        value: RecordId,
    ) -> Result<()> {
        let new_page_handle = BufferPoolManager::create_page_handle(&self.bpm).unwrap();
        let mut new_leaf_page = BPlusTreeLeafPageMut::from(new_page_handle);

        // Initialize the new leaf page which will be the new root of the B+ Tree.
        new_leaf_page.init(self.leaf_max_size);
        new_leaf_page.insert(key, value, self.comparator.as_ref())?;

        meta_page.set_root_page_id(new_leaf_page.page_id());
        Ok(())
    }

    /// Inserts the promoted middle key and page references into the parent node.
    ///
    /// This method is called after a page (either leaf or internal) has split.
    /// It inserts the promoted key and the two child page ids (left and right) into the parent node.
    ///
    /// - If the split page was the root, this method creates a new root internal page and updates
    ///   the meta page with the new root page id.
    /// - If the parent has space, the method inserts the new key and page id directly.
    /// - If the parent is full, the method recursively splits the parent and continues promotion.
    ///
    /// # Arguments
    /// - `context`: The context holding the meta page and the stack of parent pages.
    /// - `key`: The promoted key from the split.
    /// - `old_page_pid`: The page id of the left child (before split).
    /// - `new_page_pid`: The page id of the new right child (after split).
    ///
    /// # Useful Helper methods
    /// - [`BPlusTreeInternalPageMut::insert_node_after`]
    /// - [`BPlusTreeInternalPageMut::split_to_recipient_page_at`]
    /// - [`BufferPoolManager::create_page_handle`]
    fn insert_into_parent(
        &self,
        mut context: BPlusTreeContext,
        key: &BPlusTreeKey,
        old_page_pid: PageId,
        new_page_pid: PageId,
    ) -> Result<()> {
        if context.write_set.is_empty() {
            let new_root_handler = BufferPoolManager::create_page_handle(&self.bpm)?;
            let mut new_root_page = BPlusTreeInternalPageMut::from(new_root_handler);

            new_root_page.init(self.internal_max_size);
            new_root_page.set_size(2);
            new_root_page.key_array_mut()[0] = [0u8; KEY_SIZE];
            new_root_page.key_array_mut()[1] = *key;
            new_root_page.pid_array_mut()[0] = old_page_pid;
            new_root_page.pid_array_mut()[1] = new_page_pid;

            match context.meta_page {
                Some(mut meta) => meta.set_root_page_id(new_root_page.page_id()),
                None => panic!("Expected meta to be something"),
            };

            return Ok(())
        }

        let mut parent = match context.write_set.pop_back() {
            Some(BPlusTreePageMut::Internal(p)) => p,
            _ => panic!("Expected internal page"),
        };

        if parent.size() < parent.max_size() {
            parent.insert_node_after(old_page_pid, *key, new_page_pid);
            return Ok(());
        }

        // need to create a new page
        let new_internal_handle = BufferPoolManager::create_page_handle(&self.bpm)?;
        let mut new_internal_page = BPlusTreeInternalPageMut::from(new_internal_handle);
        new_internal_page.init(self.internal_max_size);

        let split_key = parent.split_to_recipient_page_at(&mut new_internal_page, key, new_page_pid, self.comparator.as_ref());

        let new_internal_pid = new_internal_page.page_id();
        let parent_pid = parent.page_id();

        self.insert_into_parent(context, &split_key, parent_pid, new_internal_pid)
    }

    /// See [`Self::coalesce_or_redistribute_leaf`]
    fn coalesce_or_redistribute_internal(
        &self,
        mut context: BPlusTreeContext,
        internal_page: BPlusTreeInternalPageMut,
    ) -> Result<()> {
        // If the current leaf is the root, no rebalancing is needed.
        if context.root_page_id == Some(internal_page.page_id()) {
            if internal_page.size() == 1 {
                let new_root_pid = internal_page.pid_array()[0];
                let old_root_pid = internal_page.page_id();
                context.meta_page.unwrap().set_root_page_id(new_root_pid);

                // Drop the internal_page handle before attempting to delete it
                drop(internal_page);

                self.bpm.write().unwrap().delete_page(old_root_pid)?;
            }

            return Ok(())
        }
        // Invariant: if this leaf is not the root, it must have a parent internal page in the write set.
        assert!(context.write_set.len() > 0);

        // Pop the parent page from the write set to update it after merging or redistributing.
        let mut parent_page = match context.write_set.pop_back() {
            Some(BPlusTreePageMut::Internal(parent)) => parent,
            _ => panic!("Expected an internal page but found something else"),
        };

        // Find the index of the leaf in the parent's child pointer array.
        let index = parent_page.find_pid_index(internal_page.page_id()).unwrap();
        let is_right_most = index == (parent_page.size() - 1);

        // Load the leaf’s left or right sibling, depending on whether it is the rightmost child.
        let (mut left, mut right) = {
            if is_right_most {
                let sibling_pid = parent_page.pid_array()[index - 1];
                let sibling = BufferPoolManager::fetch_page_mut_handle(&self.bpm, sibling_pid)?;
                let sibling_internal_page = BPlusTreeInternalPageMut::from(sibling);

                (sibling_internal_page, internal_page)
            } else {
                let sibling_pid = parent_page.pid_array()[index + 1];
                let sibling =
                    BufferPoolManager::fetch_page_mut_handle(&self.bpm, sibling_pid).unwrap();
                let sibling_internal_page = BPlusTreeInternalPageMut::from(sibling);

                (internal_page, sibling_internal_page)
            }
        };
        // The key index in the parent internal page that corresponds to the right leaf page
        let right_most_key_index = if is_right_most { index } else { index + 1 };
        let right_most_key = parent_page.key_array()[right_most_key_index];

        // Check if both leaves can fit in one page (merge case)
        // 1. Coalesces (merges) the leaf with a sibling if their combined size fits in one page.
        //    - The parent entry for the removed sibling is deleted.
        //    - The sibling's page is marked for deletion.
        //    - If the parent becomes underfull as a result, recursively re-balance the parent.

        if left.size() + right.size() <= left.max_size() {
            // apparently must always move right to left
            let pid_delete = right.page_id();
            right.move_all_to(&mut left, &right_most_key);

            // Drop the page handles before attempting to delete the page
            drop(left);
            drop(right);

            self.bpm.write().unwrap().delete_page(pid_delete)?;
            parent_page.remove_at(right_most_key_index)?;

            let parent_size = parent_page.size();
            let parent_min_size = parent_page.min_size();

            if parent_size < parent_min_size {
                self.coalesce_or_redistribute_internal(context, parent_page)?;
            }
        }
        // 2. Redistributes entries between siblings if merging is not possible.
        //    - Moves one or more entries from a sibling to balance both pages.
        //    - Updates the parent key to reflect the new split boundary.
        else {
            // Redistribution case for INTERNAL pages
            // honestly can't get the helpers to work so I adjusted the logic to a way that
            // makes more sense to me

            if is_right_most {
                // Right page wants to borrow from the left
                let last_pid = left.pid_array()[left.size() - 1];
                let new_parent_key = left.key_array()[left.size() - 1];

                // Shift right's entries to make room
                let right_size = right.size();
                // Shift pids
                for i in (0..right_size).rev() {
                    right.pid_array_mut()[i + 1] = right.pid_array()[i];
                }
                // Shift keys excluding invalid
                for i in (1..right_size).rev() {
                    right.key_array_mut()[i + 1] = right.key_array()[i];
                }

                right.key_array_mut()[1] = right_most_key;
                right.pid_array_mut()[0] = last_pid;

                right.set_size(right_size + 1);
                left.set_size(left.size() - 1);

                // Update parent separator to be the last key from left
                parent_page.key_array_mut()[right_most_key_index] = new_parent_key;
            }
            // in the else case left page wants to borrow from the right
            else {
                // Left page wants to borrow from the right
                let first_pid = right.pid_array()[0];
                let new_parent_key = right.key_array()[1];

                // Append middle_key and first_pid to left
                let left_size = left.size();
                left.key_array_mut()[left_size] = right_most_key;
                left.pid_array_mut()[left_size] = first_pid;
                left.set_size(left_size + 1);

                // Shift right's entries left to remove the first entry
                let right_size = right.size();
                for i in 1..(right_size - 1) {
                    right.key_array_mut()[i] = right.key_array()[i + 1];
                }
                for i in 0..(right_size - 1) {
                    right.pid_array_mut()[i] = right.pid_array()[i + 1];
                }
                right.set_size(right_size - 1);

                // Update parent separator to be the first key from right (before removal)
                parent_page.key_array_mut()[right_most_key_index] = new_parent_key;
            }
        }
        Ok(())
    }

    /// Handles balancing when a leaf page becomes underfull after a deletion.
    ///
    /// If a leaf page has too few entries, this method either:
    /// 1. Coalesces (merges) the leaf with a sibling if their combined size fits in one page.
    ///    - The parent entry for the removed sibling is deleted.
    ///    - The sibling's page is marked for deletion.
    ///    - If the parent becomes underfull as a result, recursively re-balance the parent.
    /// 2. Redistributes entries between siblings if merging is not possible.
    ///    - Moves one or more entries from a sibling to balance both pages.
    ///    - Updates the parent key to reflect the new split boundary.
    ///
    /// If the leaf page is the root and becomes empty, the tree is collapsed by setting the root
    /// to an invalid state.
    ///
    /// # Arguments
    /// * `context` - Holds meta and parent pages to assist with rebalancing.
    /// * `leaf_page` - The underfull leaf page being rebalanced.
    ///
    /// # Useful Helper methods
    /// - [`BufferPoolManager::fetch_page_mut_handle`]
    /// - [`BPlusTreeInternalPageMut::move_all_to`]
    /// - [`BPlusTreeLeafPageMut::move_last_to_front_of`]
    /// - [`BPlusTreeLeafPageMut::move_first_to_end_of`]
    /// - [`BPlusTreeInternalPageMut::min_size`]
    /// - [`BPlusTreeInternalPageMut::key_array_mut`]
    fn coalesce_or_redistribute_leaf(
        &self,
        mut context: BPlusTreeContext,
        leaf_page: BPlusTreeLeafPageMut,
    ) -> Result<()> {
        // If the current leaf is the root, no rebalancing is needed.
        if context.root_page_id == Some(leaf_page.page_id()) {
            return Ok(());
        }
        // Invariant: if this leaf is not the root, it must have a parent internal page in the write set.
        assert!(context.write_set.len() > 0);

        // Pop the parent page from the write set to update it after merging or redistributing.
        let mut parent_page = match context.write_set.pop_back() {
            Some(BPlusTreePageMut::Internal(parent)) => parent,
            _ => panic!("Expected an internal page but found something else"),
        };

        // Find the index of the leaf in the parent's child pointer array.
        let index = parent_page.find_pid_index(leaf_page.page_id()).unwrap();
        let is_right_most = index == (parent_page.size() - 1);

        // Load the leaf’s left or right sibling, depending on whether it is the rightmost child.
        let (mut left, mut right) = {
            if is_right_most {
                let sibling_pid = parent_page.pid_array()[index - 1];
                let sibling = BufferPoolManager::fetch_page_mut_handle(&self.bpm, sibling_pid)?;
                let sibling_leaf_page = BPlusTreeLeafPageMut::from(sibling);

                (sibling_leaf_page, leaf_page)
            } else {
                let sibling_pid = parent_page.pid_array()[index + 1];
                let sibling =
                    BufferPoolManager::fetch_page_mut_handle(&self.bpm, sibling_pid).unwrap();
                let sibling_leaf_page = BPlusTreeLeafPageMut::from(sibling);

                (leaf_page, sibling_leaf_page)
            }
        };
        // The key index in the parent internal page that corresponds to the right leaf page
        let right_most_key_index = if is_right_most { index } else { index + 1 };

        // Check if both leaves can fit in one page (merge case)
        // 1. Coalesces (merges) the leaf with a sibling if their combined size fits in one page.
        //    - The parent entry for the removed sibling is deleted.
        //    - The sibling's page is marked for deletion.
        //    - If the parent becomes underfull as a result, recursively re-balance the parent.
        if left.size() + right.size() <= left.max_size() - 1 {
            let pid_delete = right.page_id();
            right.move_all_to(&mut left);

            // Drop the page handles before attempting to delete the page
            drop(left);
            drop(right);

            self.bpm.write().unwrap().delete_page(pid_delete)?;
            parent_page.remove_at(right_most_key_index)?;

            let parent_size = parent_page.size();
            let parent_min_size = parent_page.min_size();

            if parent_size < parent_min_size {
                self.coalesce_or_redistribute_internal(context, parent_page)?;
            }
        }
        // 2. Redistributes entries between siblings if merging is not possible.
        //    - Moves one or more entries from a sibling to balance both pages.
        //    - Updates the parent key to reflect the new split boundary.
        else {
            // Redistribution case
            // in the right most case right page wants to borrow from the left
            if is_right_most {
                left.move_last_to_front_of(&mut right);
            }
            // in the else case left page wants to borrow from the right
            else {
                right.move_first_to_end_of(&mut left);
            }

            // in both cases it's the first key of the right that becomes the new parent
            parent_page.key_array_mut()[right_most_key_index] = right.key_array()[0];
        }
        Ok(())
    }

    /// This is a helper method that is provided to you.
    ///
    /// Traverses the B+ Tree from the root down to the target leaf page for insert or delete.
    ///
    /// This method tracks all pages visited in a BPlusTreeContext for locking and structural updates.
    ///
    /// For insertion:
    /// - If the tree is empty, calls start_new_tree to initialize the root and returns None.
    ///
    /// For both insertion and deletion:
    /// - Walks down the tree using lookup until it reaches a leaf.
    /// - Uses is_safe to check if the current subtree can safely be unlocked early.
    /// - Pushes visited internal pages onto the write_set for later updates.
    ///
    /// Returns the traversal context and the final leaf page handle.
    fn traverse_down_to_leaf(
        &self,
        key: &BPlusTreeKey,
        val: Option<RecordId>,
        access: FindLeafOption,
    ) -> Option<(BPlusTreeContext, PageFrameMutHandle)> {
        let mut context = BPlusTreeContext::new();
        let meta_page_handle =
            BufferPoolManager::fetch_page_mut_handle(&self.bpm, self.meta_page_id).unwrap();
        let mut meta_page = BPlusTreeMetaPageMut::from(meta_page_handle);
        match access {
            FindLeafOption::Insert => {
                if meta_page.root_page_id().is_none() {
                    self.start_new_tree(&mut meta_page, key, val.unwrap());
                    return None;
                }
                context.root_page_id = Some(meta_page.root_page_id().unwrap());
                context.meta_page = Some(meta_page);
            }
            FindLeafOption::Delete => {
                context.root_page_id = Some(meta_page.root_page_id()?);
                context.meta_page = Some(meta_page);
            }
            FindLeafOption::Lookup => {
                context.root_page_id = Some(meta_page.root_page_id()?);
                context.meta_page = Some(meta_page);
            }
        }

        let mut cur_handle =
            BufferPoolManager::fetch_page_mut_handle(&self.bpm, context.root_page_id.unwrap())
                .unwrap();
        if BPlusTreeContext::is_safe(BPlusTreePageHeader::header_mut(&cur_handle), true, access) {
            context.unlock_all();
        }

        while BPlusTreePageHeader::get_page_type_from_frame_mut(&cur_handle)
            != BPlusTreePageType::LeafPage
        {
            let internal_page = BPlusTreeInternalPageMut::from(cur_handle);
            let child_pid = internal_page.lookup(&key, self.comparator.as_ref());

            context
                .write_set
                .push_back(BPlusTreePageMut::Internal(internal_page));

            cur_handle = BufferPoolManager::fetch_page_mut_handle(&self.bpm, child_pid).unwrap();

            if BPlusTreeContext::is_safe(
                BPlusTreePageHeader::header_mut(&cur_handle),
                false,
                access,
            ) {
                context.unlock_all();
            }
        }
        Some((context, cur_handle))
    }

    fn make_empty_iterator(&self, end: Option<&BPlusTreeKey>) -> BPlusTreeRangeIterator {
        return BPlusTreeRangeIterator {
            bpm: Arc::clone(&self.bpm),
            current_page_id: INVALID_PAGE_ID,
            offset: 0,
            end_key: end.copied()
        }
    }
}

impl BPlusTreeIndex for BPlusTreeIndexImpl {
    /// Traverses the tree and retrieves the record id corresponding to a tuple with the given
    /// search key, if one exists. Otherwise, returns None.
    ///
    /// Useful helper methods:
    /// - [`BufferPoolManager::fetch_page_handle`]
    /// - [`BPlusTreeLeafPageMut::lookup`]
    /// - [`BPlusTreeInternalPageRef::lookup`]
    /// - [`BPlusTreePageHeader::get_page_type_from_frame_ref`]
    fn get(&self, key: &BPlusTreeKey) -> Option<RecordId> {
        let meta_handle = BufferPoolManager::fetch_page_handle(&self.bpm, self.meta_page_id).ok()?;
        let meta_page = BPlusTreeMetaPageRef::from(meta_handle);
        let root_pid = meta_page.root_page_id()?;

        let mut cur_handle = BufferPoolManager::fetch_page_handle(&self.bpm, root_pid).ok()?;

        while BPlusTreePageHeader::get_page_type_from_frame_ref(&cur_handle) != BPlusTreePageType::LeafPage {
            let internal_page = BPlusTreeInternalPageRef::from(cur_handle);
            let child_pid = internal_page.lookup(key, self.comparator.as_ref());

            cur_handle = BufferPoolManager::fetch_page_handle(&self.bpm, child_pid).ok()?;
        }

        let leaf_page = BPlusTreeLeafPageRef::from(cur_handle);
        leaf_page.lookup(key, self.comparator.as_ref())
    }

    /// Inserts a key-value pair into the tree.
    ///
    /// Useful helper methods:
    /// - [`BufferPoolManager::create_page_handle`]
    /// - [`BPlusTreeLeafPageMut::move_half_to`]
    /// - [`Self::insert_into_parent`]
    fn insert(&self, key: &BPlusTreeKey, val: RecordId) -> Result<()> {
        // Traverse the tree
        let (context, cur_handle) =
            match self.traverse_down_to_leaf(key, Some(val), FindLeafOption::Insert) {
                Some(pair) => pair,
                None => return Ok(()),
            };
        let mut leaf_page = BPlusTreeLeafPageMut::from(cur_handle);

        if leaf_page.size() < leaf_page.max_size() {
            leaf_page.insert(key, val, self.comparator.as_ref())?;

            return Ok(())
        }

        // now we need to split
        let new_leaf_handle = BufferPoolManager::create_page_handle(&self.bpm)?;
        let mut new_leaf_page = BPlusTreeLeafPageMut::from(new_leaf_handle);

        let old_pid = leaf_page.page_id();
        let new_pid = new_leaf_page.page_id();

        // Split the page first (moves half the entries to new page)
        leaf_page.move_half_to(&mut new_leaf_page, new_pid);

        // Now insert into the appropriate page based on key comparison
        if self.comparator.compare(key, &new_leaf_page.key_array()[0]) == std::cmp::Ordering::Less {
            // Key belongs in the left (old) page
            leaf_page.insert(key, val, self.comparator.as_ref())?;
        } else {
            // Key belongs in the right (new) page
            new_leaf_page.insert(key, val, self.comparator.as_ref())?;
        }

        let promoted = new_leaf_page.key_array()[0];
        self.insert_into_parent(context, &promoted, old_pid, new_pid)?;

        Ok(())
    }

    /// Removes a key-value pair from the tree. If the key is found, returns the associated record
    /// id of the corresponding tuple containing that key. Otherwise, returns None.
    ///
    /// Useful helper methods:
    /// - [`BPlusTreeLeafPageMut::remove_and_delete_record`]
    /// - [`Self::coalesce_or_redistribute_internal`]
    /// - [`Self::coalesce_or_redistribute_leaf`]
    /// - [`BufferPoolManager::delete_page`]
    /// - [`BPlusTreeMetaPageMut::set_root_page_id`]
    fn remove(&self, key: &BPlusTreeKey) -> Option<RecordId> {
        let (mut context, cur_handle) =
            self.traverse_down_to_leaf(key, None, FindLeafOption::Delete)?;
        let mut leaf_page = BPlusTreeLeafPageMut::from(cur_handle);

        let removed_val = leaf_page.remove_and_delete_record(key, self.comparator.as_ref())?;

        if context.root_page_id != Some(leaf_page.page_id())
            && leaf_page.size() < leaf_page.min_size() {
            // do expect because it really is an error
            self.coalesce_or_redistribute_leaf(context, leaf_page)
                .expect("Failed to rebalance leaf after deletion");
        }

        Some(removed_val)
    }

    /// Returns an iterator over key-value pairs in sorted order, bounded by optional start and end keys.
    ///
    /// Useful helper methods:
    /// - [`Self::traverse_down_to_leaf`]
    /// - [`BPlusTreeLeafPageMut::find_key_index`]
    fn range(
        &self,
        start: Option<&BPlusTreeKey>,
        end: Option<&BPlusTreeKey>,
    ) -> BPlusTreeRangeIterator {
        let meta_handle = match BufferPoolManager::fetch_page_handle(&self.bpm, self.meta_page_id) {
            Ok(val)  => val,
            Err(_) => return self.make_empty_iterator(end)
        };
        let meta_page = BPlusTreeMetaPageRef::from(meta_handle);
        let root_pid = match meta_page.root_page_id() {
            Some(pid) => pid,
            None => {
                return self.make_empty_iterator(end)
            }
        };

        let mut cur_handle = match BufferPoolManager::fetch_page_handle(&self.bpm, root_pid) {
            Ok(val) => val,
            Err(_) => return self.make_empty_iterator(end)
        };
        let leaf_page = {
            while BPlusTreePageHeader::get_page_type_from_frame_ref(&cur_handle) != BPlusTreePageType::LeafPage {
                let internal_page = BPlusTreeInternalPageRef::from(cur_handle);

               // loop structure is the same for both but in the None case you always take the left most pid 
                let child_pid = match start {
                    Some(val) => {
                        internal_page.lookup(val, self.comparator.as_ref())
                    }
                    None => {
                        internal_page.pid_array()[0]
                    }
                };

                cur_handle = match BufferPoolManager::fetch_page_handle(&self.bpm, child_pid) {
                    Ok(val) => val,
                    Err(_) => return self.make_empty_iterator(end)
                };
            }

            BPlusTreeLeafPageRef::from(cur_handle)
        };
        
        let offset = match start {
            Some(val) => {
                match leaf_page.find_key_index(val, self.comparator.as_ref()) {
                    Some(idx) => idx,
                    None => {
                        // Key doesn't exist - find first key > val
                        // wanted to use the insert position function but that is private
                        let size = leaf_page.size();
                        let mut idx = 0;
                        while idx < size {
                            if self.comparator.compare(&leaf_page.key_array()[idx], val) == Ordering::Greater {
                                break;
                            }
                            idx += 1;
                        }
                        idx  // Will be size if all keys <= val
                    }
                }
            },
            None => 0
        };

        return BPlusTreeRangeIterator {
            bpm: Arc::clone(&self.bpm),
            current_page_id: leaf_page.page_id(),
            offset: offset,
            end_key: end.copied()
        }
    }
}

impl Iterator for BPlusTreeRangeIterator {
    type Item = (BPlusTreeKey, RecordId);

    /// Advances the iterator and returns the next key-value pair.
    ///
    /// Returns `None` if:
    /// - The end of the tree is reached.
    /// - The next key exceeds the optional `end_key` bound.
    ///
    /// This method internally loads pages as needed using the buffer pool manager.
    fn next(&mut self) -> Option<Self::Item> {
       loop {
            // we'll use this as a sentry to tell us we're at the end of the tree
            if self.current_page_id == INVALID_PAGE_ID {
                return None;
            }

            let handle = match BufferPoolManager::fetch_page_handle(&self.bpm, self.current_page_id) {
                Ok(handle) => handle,
                Err(_) => return None,
            };

            let leaf_page = BPlusTreeLeafPageRef::from(handle);

            // Check if we've reached the end of the current page
            if self.offset >= leaf_page.size() {
                self.current_page_id = leaf_page.next_page_id().unwrap_or(INVALID_PAGE_ID);
                self.offset = 0;
                continue;  // Go to next iteration to fetch the new page
            }

            let current_key_idx = self.offset;
            self.offset += 1;

            let current_key = leaf_page.key_array()[current_key_idx];
            let current_record_id = leaf_page.rid_array()[current_key_idx];

            // Check if we're past the end boundary
            if let Some(end) = self.end_key {
                if current_key > end {
                    return None;
                }
            }

            // Return the current key-value pair
            return Some((current_key, current_record_id));
       }
    }
}


/// An iterator that performs a forward range scan over key-value pairs in the B+ Tree.
///
/// The iterator starts at a specified leaf page and offset, and optionally stops at a specified end key.
/// It advances through the leaf pages using the `next_page_id` pointer until no more entries remain
/// or the end key is reached.
pub struct BPlusTreeRangeIterator {
    /// Shared reference to the buffer pool manager for page access.
    bpm: Arc<RwLock<BufferPoolManager>>,
    /// The current leaf page being scanned.
    current_page_id: PageId,
    /// The current offset within the leaf page's key-value arrays.
    offset: usize,
    /// Optional end key that bounds the iteration.
    end_key: Option<BPlusTreeKey>,
}

impl BPlusTreeRangeIterator {
    /// Creates a new range iterator starting at `page_id` and `offset`,
    /// optionally stopping at `end_key` if provided.
    ///
    /// # Arguments
    /// - `bpm`: Shared reference to the buffer pool manager.
    /// - `page_id`: The starting leaf page id.
    /// - `offset`: The starting offset within the page.
    /// - `end_key`: Optional inclusive upper bound for iteration.
    pub fn new(
        bpm: Arc<RwLock<BufferPoolManager>>,
        page_id: PageId,
        offset: usize,
        end_key: Option<BPlusTreeKey>,
    ) -> Self {
        Self {
            bpm,
            current_page_id: page_id,
            offset,
            end_key,
        }
    }
}