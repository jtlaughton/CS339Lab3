use crate::schema::{RecordId, Schema};
use crate::tuple::Tuple;
use rustdb_error::Result;
use std::collections::HashMap;
use std::sync::Arc;

pub type TableId = u32;
pub type IndexId = u32;

/// Stores metadata about a given table in a DBMS.
pub struct TableInfo {
    id: TableId,
    name: String,
    schema: Schema,
}

/// A catalog of relevant information and references to objects relevant to the query execution.
/// Designed for use by executors in the execution engine of a DBMS, providing a centralized API
/// for table creation and table lookup.
pub struct Catalog<S: StorageApi> {
    /// The storage engine used by our DBMS.
    storage: Arc<S>,
    /// Maps table id -> table metadata.
    tables: HashMap<TableId, TableInfo>,
    /// Maps table name -> table id.
    table_names: HashMap<String, TableId>,
    /// The next `TableId` to be used.
    next_table_id: std::sync::atomic::AtomicU32,
}

impl<S: StorageApi> Catalog<S> {
    /// Instantiates a system catalog, given a reference to the storage engine that we'll use.
    pub fn new(storage: Arc<S>) -> Self {
        Self {
            storage,
            tables: HashMap::new(),
            table_names: HashMap::new(),
            next_table_id: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Creates a new table with the given name and schema.
    ///
    /// NOTE: We do not allow more than one table to share the same table name!
    pub fn create_table(&mut self, name: String, schema: Schema) -> &TableInfo {
        assert!(
            !self.table_names.contains_key(&name),
            "Table names must be unique."
        );

        let new_table_id = {
            // Generate the id for the new table, and map the table name to this id.
            let id = self
                .next_table_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.table_names.insert(name.clone(), id);

            // Update the table metadata map.
            let info = TableInfo { id, name, schema };
            self.tables.insert(id, info);
            id
        };
        self.tables.get(&new_table_id).unwrap()
    }

    /// Fetches the metadata for the table with given id, if one exists.
    pub fn table_with_id(&self, id: TableId) -> Option<&TableInfo> {
        self.tables.get(&id)
    }

    /// Fetches the metadata for the table with given name, if one exists.
    pub fn table_with_name(&self, name: &str) -> Option<&TableInfo> {
        let id = self.table_names.get(name)?;
        self.tables.get(id)
    }

    /// Fetches an iterator over table with the given id, if one exists.
    pub fn table_iter(&self, id: TableId) -> Option<S::ScanIterator> {
        self.storage.scan(id).map_or(None, |iter| Some(iter))
    }
}

/// An iterator that emits tuples sequentially scanned from a table.
///
/// NOTE: This iterator returns items that are owned values instead of references. This design
/// prevents dependencies on the storage engine’s memory model, ensuring tuples remain valid even
/// if the underlying storage is modified (e.g., pages are evicted). Implementing types should
/// handle value copying appropriately.
pub trait ScanIterator: Iterator<Item = Result<(RecordId, Tuple)>> {}
/// Blanket implementation of ScanIterator for any `T` satisfying the trait bound.
impl<T: Iterator<Item = Result<(RecordId, Tuple)>>> ScanIterator for T {}

/// Should be implemented by the storage engine we're using to enable an access interface between
/// the execution engine using this Catalog and its corresponding storage engine.
pub trait StorageApi {
    /// The iterator emitted by [`StorageApi::scan`]. This associated type allows the type
    /// implementing this trait to return a specific iterator type without requiring dynamic
    /// dispatch; thus, the Rust compiler can monomorphize [`StorageApi::scan`] at compile time,
    /// avoiding both the vtable lookup required in [`StorageApi::scan_dyn`] and subsequent the
    /// heap allocation occurring with the [`Box<dyn ScanIterator>`] return type.
    type ScanIterator: ScanIterator
    where
        Self: Sized;

    /// Creates a table with the given name and id.
    fn create_table(&self, table_id: TableId, name: &str) -> Result<&TableInfo>;

    /// Retrieves a tuple, with record id `rid`, from the table with corresponding id `table_id`.
    fn get_tuple(&self, table_id: TableId, rid: RecordId) -> Result<Tuple>;

    /// Deletes a tuple, with record id `rid`, from the table with corresponding id `table_id`.
    fn delete_tuple(&self, table_id: TableId, rid: RecordId) -> Result<()>;

    /// Inserts the given tuple into the table with corresponding id `table_id`, returning the
    /// newly inserted tuple's record id.
    fn insert_tuple(&self, table_id: TableId, tuple: &Tuple) -> Result<RecordId>;

    /// Retrieves an iterator that emits tuples from a table via sequential scan.
    fn scan(&self, table_id: TableId) -> Result<Self::ScanIterator>
    where
        Self: Sized;

    /// [`StorageApi::scan`], but can be used from trait objects. This iterator uses dynamic
    /// dispatch, which incurs a runtime performance penalty.
    fn scan_dyn(&self, table_id: TableId) -> Result<Box<dyn ScanIterator>>;
}
