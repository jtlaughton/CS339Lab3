use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::{
    buffer_pool::BufferPoolManager,
    heap::{table_heap::TableHeap, table_tuple_iterator::TableTupleIterator},
    Result,
};
use rustdb_catalog::{
    catalog::{self, StorageApi},
    schema,
    tuple::Tuple,
};
use rustdb_error::Error;

/// A storage engine that implements StorageApi using a table heap.
/// It maintains a mapping from table IDs to table heaps (each wrapped in an RwLock).
pub struct StorageEngine {
    bpm: Arc<RwLock<BufferPoolManager>>,
    // Each table heap is now wrapped in an RwLock for internal synchronization.
    tables: RwLock<HashMap<catalog::TableId, Arc<RwLock<TableHeap>>>>,
}

impl StorageEngine {
    /// Creates a new StorageEngine given a BufferPoolManager.
    pub fn new(bpm: Arc<RwLock<BufferPoolManager>>) -> Self {
        Self {
            bpm: Arc::clone(&bpm),
            tables: RwLock::new(HashMap::new()),
        }
    }
}

impl StorageApi for StorageEngine {
    /// The iterator type for scanning tuples in a table.
    type ScanIterator = TableTupleIterator;

    /// Creates a new table.
    ///
    /// In a full system this would create a new table heap and a catalog entry.
    /// Here we simply create a new TableHeap, wrap it in an RwLock, and store it in our map.
    fn create_table(&self, table_id: catalog::TableId, name: &str) -> Result<&catalog::TableInfo> {
        let mut tables = self.tables.write().unwrap();
        if tables.contains_key(&table_id) {
            return Err(Error::InvalidInput("Table already exists".to_string()));
        }
        let table_heap = TableHeap::new(name, self.bpm.clone());
        // Wrap the TableHeap in an RwLock.
        tables.insert(table_id, Arc::new(RwLock::new(table_heap)));
        todo!("Return a reference to the newly created TableInfo")
    }

    /// Retrieves a tuple given its record id.
    fn get_tuple(&self, table_id: catalog::TableId, rid: schema::RecordId) -> Result<Tuple> {
        let tables = self.tables.read().unwrap();
        let table_heap_lock = tables
            .get(&table_id)
            .ok_or_else(|| Error::InvalidInput("Table not found".to_string()))?;
        // Acquire a read lock on the table heap.
        let table_heap = table_heap_lock.read().unwrap();
        // TableHeap::get_tuple returns a (TupleMetadata, Tuple) pair.
        let (_meta, tuple) = table_heap.get_tuple(&rid.into())?;
        Ok(tuple)
    }

    /// Deletes a tuple given its record id.
    fn delete_tuple(&self, table_id: catalog::TableId, rid: schema::RecordId) -> Result<()> {
        let tables = self.tables.read().unwrap();
        let table_heap_lock = tables
            .get(&table_id)
            .ok_or_else(|| Error::InvalidInput("Table not found".to_string()))?;
        // Acquire a write lock to modify the table heap.
        let table_heap = table_heap_lock.write().unwrap();
        table_heap.delete_tuple(&rid.into())?;
        Ok(())
    }

    /// Inserts a tuple into the specified table.
    fn insert_tuple(&self, table_id: catalog::TableId, tuple: &Tuple) -> Result<schema::RecordId> {
        let mut tables = self.tables.write().unwrap();
        let table_heap_lock = tables
            .get_mut(&table_id)
            .ok_or_else(|| Error::InvalidInput("Table not found".to_string()))?;
        // Acquire a write lock for insertion.
        let mut table_heap = table_heap_lock.write().unwrap();
        let rid = table_heap.insert_tuple(tuple)?;
        Ok(rid.into())
    }

    /// Returns an iterator over all tuples in the specified table.
    fn scan(&self, table_id: catalog::TableId) -> Result<Self::ScanIterator>
    where
        Self: Sized,
    {
        let tables = self.tables.read().unwrap();
        let table_heap_lock = tables
            .get(&table_id)
            .ok_or_else(|| Error::InvalidInput("Table not found".to_string()))?;
        Ok(TableTupleIterator::new(
            self.bpm.clone(),
            table_heap_lock.clone(),
        ))
    }

    /// Returns a dynamic iterator over all tuples in the specified table.
    fn scan_dyn(&self, table_id: catalog::TableId) -> Result<Box<dyn catalog::ScanIterator>> {
        Ok(Box::new(self.scan(table_id)?))
    }
}
