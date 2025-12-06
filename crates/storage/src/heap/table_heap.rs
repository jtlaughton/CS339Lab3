use std::sync::{Arc, RwLock};

use rustdb_catalog::tuple::Tuple;
use rustdb_error::Error;

use crate::page::INVALID_PAGE_ID;
use crate::{
    buffer_pool::BufferPoolManager,
    page::table_page::{TablePageMut, TablePageRef, TupleMetadata},
    record_id::RecordId,
    typedef::PageId,
    Result,
};

pub struct TableHeap {
    table_name: String,
    page_cnt: u32,
    bpm: Arc<RwLock<BufferPoolManager>>,
    first_page_id: PageId,
    last_page_id: PageId,
}

impl TableHeap {
    /// Create a new table heap. A new root page is allocated from the buffer pool.
    pub fn new(name: &str, bpm: Arc<RwLock<BufferPoolManager>>) -> TableHeap {
        let bpm_clone = bpm.clone();
        let root_page = BufferPoolManager::create_page_handle(&bpm_clone)
            .expect("BufferPoolManager creation should not fail");

        let first_page_id = root_page.page_id();

        let mut table_page = TablePageMut::from(root_page);
        table_page.init_header(INVALID_PAGE_ID);

        TableHeap {
            table_name: name.to_string(),
            page_cnt: 1,
            bpm,
            first_page_id,
            last_page_id: first_page_id
        }
    }

    /// Retrieve a tuple given its record id.
    pub fn get_tuple(&self, rid: &RecordId) -> Result<(TupleMetadata, Tuple)> {
        let table_page = {
            let page_handle = BufferPoolManager::fetch_page_handle(&self.bpm, rid.page_id())?;
            TablePageRef::from(page_handle)
        };

        table_page.get_tuple(rid)
    }

    /// Delete a tuple given its record id, returning the deleted tuple (and its metadata).
    pub fn delete_tuple(&self, rid: &RecordId) -> Result<(TupleMetadata, Tuple)> {
        let (og_meta, og_tuple) = self.get_tuple(rid)?;

        let mut table_page = {
            let page_handle = BufferPoolManager::fetch_page_mut_handle(&self.bpm, rid.page_id())?;
            TablePageMut::from(page_handle)
        };

        let new_meta = TupleMetadata::new(true);
        table_page.update_tuple_metadata(rid, new_meta)?;

        Ok((og_meta, og_tuple))
    }

    /// Insert a tuple into the table heap.
    pub fn insert_tuple(&mut self, tuple: &Tuple) -> Result<RecordId> {
        // For a newly inserted tuple the metadata is by default not deleted
        let metadata = TupleMetadata::new(false);

        // Try to fetch a mutable handle for the current last page.
        let mut current_table_page = {
            let page_handle =
                BufferPoolManager::fetch_page_mut_handle(&self.bpm, self.last_page_id)?;
            // `TablePage` takes ownership of the page handle, so the page handle won't be dropped
            // (and thus the frame won't be unpinned in the buffer pool) until this table page
            // goes out of scope.
            TablePageMut::from(page_handle)
        };

        // Try inserting the tuple into the current page.
        match current_table_page.insert_tuple(&metadata, tuple) {
            // It worked!
            Ok(rid) => Ok(rid),
            // Uh oh, there isn’t enough free space in the current page...
            Err(Error::OutOfBounds) => {
                // Allocate a new page.
                let mut new_table_page =
                    TablePageMut::from(BufferPoolManager::create_page_handle(&self.bpm)?);
                let new_page_id = new_table_page.page_id();

                // Update the current page’s header to point to the new page.
                current_table_page.set_next_page_id(new_page_id);

                // Initialize the new page (its header’s next_page_id is set to INVALID_PAGE_ID).
                new_table_page.init_header(INVALID_PAGE_ID);

                // Try inserting the tuple into the new page.
                let rid = new_table_page.insert_tuple(&metadata, tuple)?;
                // Update the table heap’s bookkeeping.
                self.last_page_id = new_page_id;
                self.page_cnt += 1;

                Ok(rid)
            }
            Err(e) => Err(e),
        }
    }

    pub(crate) fn first_page_id(&self) -> PageId {
        self.first_page_id
    }
}

#[cfg(test)]
mod tests {
    use rustdb_catalog::tuple::Tuple;
    use serial_test::serial;

    use crate::replacer::lru_k_replacer::LrukReplacer;
    use std::sync::{Arc, Mutex, RwLock};

    use crate::disk::disk_manager::DiskManager;
    use crate::heap::table_heap::TableHeap;
    use crate::page::table_page::{TABLE_PAGE_HEADER_SIZE, TUPLE_INFO_SIZE};
    use crate::page::PAGE_SIZE;
    use crate::{buffer_pool::BufferPoolManager, Result};

    pub fn get_bpm_with_pool_size(pool_size: usize) -> BufferPoolManager {
        let disk_manager = Arc::new(Mutex::new(DiskManager::new("test.db").unwrap()));
        let replacer = Box::new(LrukReplacer::new(5));
        BufferPoolManager::new(pool_size, disk_manager, replacer)
    }

    fn get_bpm_arc_with_pool_size(pool_size: usize) -> Arc<RwLock<BufferPoolManager>> {
        Arc::new(RwLock::new(get_bpm_with_pool_size(pool_size)))
    }

    /// Test that we can insert a tuple into the table heap and then retrieve it correctly.
    #[test]
    #[serial]
    fn test_table_heap_insert_and_get() -> Result<()> {
        let bpm = get_bpm_arc_with_pool_size(10);

        let mut table_heap = TableHeap::new("table", bpm.clone());

        let tuple_data = vec![10, 20, 30, 40, 50];
        let tuple = Tuple::new(tuple_data.clone().into());

        let rid = table_heap.insert_tuple(&tuple)?;
        let (meta, retrieved_tuple) = table_heap.get_tuple(&rid)?;
        assert_eq!(retrieved_tuple.data(), tuple_data.as_slice());
        assert!(!meta.is_deleted());

        Ok(())
    }

    /// Test that a tuple insertion that would overflow the current page
    /// triggers allocation of a new page and that both tuples are correctly stored.
    #[test]
    #[serial]
    fn test_table_heap_new_page_allocation() -> Result<()> {
        let bpm = get_bpm_arc_with_pool_size(10);

        let mut table_heap = TableHeap::new("table", bpm.clone());

        // Create and insert a huge tuple that nearly fills the page.
        let huge_tuple_size = PAGE_SIZE - TABLE_PAGE_HEADER_SIZE - TUPLE_INFO_SIZE - 5;
        let huge_tuple_data = vec![1; huge_tuple_size];
        let huge_tuple = Tuple::new(huge_tuple_data.clone().into());
        let rid1 = table_heap.insert_tuple(&huge_tuple)?;

        // Insert another tuple. This insertion should detect insufficient space in the
        // current page and cause a new page to be allocated.
        let small_tuple_data = vec![2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5];
        let small_tuple = Tuple::new(small_tuple_data.clone().into());
        let rid2 = table_heap.insert_tuple(&small_tuple)?;

        // Verify that the two record IDs have different page ids.
        assert_ne!(rid1.page_id(), rid2.page_id());

        // Retrieve both tuples and verify their data.
        let (_meta1, retrieved_huge) = table_heap.get_tuple(&rid1)?;
        let (_meta2, retrieved_small) = table_heap.get_tuple(&rid2)?;
        assert_eq!(retrieved_huge.data(), huge_tuple_data.as_slice());
        assert_eq!(retrieved_small.data(), small_tuple_data.as_slice());

        Ok(())
    }

    #[test]
    #[serial]
    fn test_tuple_deletion() {
        let bpm = get_bpm_arc_with_pool_size(2);
        let mut table_heap = TableHeap::new("table", bpm.clone());

        // Insert tuples into table heap
        let tuples: Vec<Tuple> = vec![
            Tuple::new(vec![10, 20, 30].into()),
            Tuple::new(vec![40, 50, 60].into()),
            Tuple::new(vec![70, 80, 90].into()),
        ];

        let mut rids = Vec::new();
        for tuple in &tuples {
            let rid = table_heap.insert_tuple(tuple).unwrap();
            rids.push(rid);
        }

        // Delete the second tuple
        let (meta_deleted, tuple_deleted) = table_heap.delete_tuple(&rids[1]).unwrap();
        assert_eq!(meta_deleted.is_deleted(), false);
        assert_eq!(table_heap.get_tuple(&rids[1]).unwrap().0.is_deleted(), true);
        assert_eq!(tuple_deleted.data(), tuples[1].data());

        // Verify deleted tuple metadata is marked correctly
        let (meta_after, _) = table_heap.get_tuple(&rids[1]).unwrap();
        assert!(meta_after.is_deleted(), "Tuple should be marked as deleted");

        // Verify other tuples are unaffected
        for (i, rid) in rids.iter().enumerate() {
            if i != 1 {
                let (meta, tuple) = table_heap.get_tuple(rid).unwrap();
                assert!(!meta.is_deleted(), "Tuple {} should not be deleted", i);
                assert_eq!(tuple.data(), tuples[i].data(), "Tuple {} data mismatch", i);
            }
        }

        // Attempt to delete the same tuple again
        let delete_again_result = table_heap.delete_tuple(&rids[1]);
        assert!(
            delete_again_result.is_ok(),
            "Deleting an already deleted tuple should succeed"
        );

        // Final check to ensure no unintended side effects.
        for (i, rid) in rids.iter().enumerate() {
            let (meta, tuple) = table_heap.get_tuple(rid).unwrap();
            if i == 1 {
                assert!(meta.is_deleted(), "Tuple {} should remain deleted", i);
            } else {
                assert!(
                    !meta.is_deleted(),
                    "Tuple {} should still not be deleted",
                    i
                );
                assert_eq!(
                    tuple.data(),
                    tuples[i].data(),
                    "Tuple {} data integrity check failed",
                    i
                );
            }
        }
    }
}
