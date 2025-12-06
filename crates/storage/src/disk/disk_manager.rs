use crate::typedef::PageId;
use crate::Result;
use bytes::{Bytes, BytesMut};
use fs2::FileExt;
use rustdb_error::{errdata, Error};
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

pub(crate) const DATA_DIR: &str = "src/disk/data/";
const PAGE_SIZE_BYTES: usize = 4096;

const EMPTY_BUFFER: &[u8] = &[0; PAGE_SIZE_BYTES];

#[derive(Debug)]
pub struct DiskManager {
    file: RefCell<std::fs::File>,
    /// The maximum capacity (in pages) that the file can hold before we resize it.
    page_capacity: usize,
    /// Tracks the highest page_id allocated so far.
    last_allocated_pid: PageId,
    /// Map from page_id -> file offset
    pages: HashMap<PageId, u64>,
    /// Free file offsets to reuse for future page allocations.
    free_slots: VecDeque<u64>,
}

impl DiskManager {
    /// Creates a new disk manager for the given database file `filename`.
    /// The file is truncated and locked exclusively at creation.
    pub(crate) fn new(filename: &str) -> Result<Self> {
        let path = Path::new(DATA_DIR).join(filename);

        // Open or create the file, truncating it
        let file = std::fs::OpenOptions::new()
            .write(true)
            .read(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| Error::IO(format!("Unable to open file {}: {}", path.display(), e)))?;

        // Acquire an exclusive lock on the file at creation
        file.lock_exclusive()
            .map_err(|e| Error::IO(format!("Failed to acquire exclusive file lock: {}", e)))?;

        // Build the DiskManager struct
        let mut dm = Self {
            file: RefCell::new(file),
            page_capacity: 32, // Start with 32 as the default capacity
            last_allocated_pid: 0,
            pages: HashMap::new(),
            free_slots: VecDeque::new(),
        };

        // Initialize the file with enough space for `page_capacity + 1` pages
        dm.resize_file()?;

        Ok(dm)
    }

    /// Allocate a new page_id and a file offset for storing it.
    pub fn allocate_page(&mut self) -> Result<PageId> {
        self.last_allocated_pid += 1;
        let pid = self.last_allocated_pid;

        // Find or create an offset for the page
        let new_offset = self.allocate_offset()?;
        // Record pid -> offset
        self.pages.insert(pid, new_offset);
        // Initialize the page with empty data
        self.write(pid, EMPTY_BUFFER)?;

        Ok(pid)
    }

    /// Deallocates a page and adds its offset to the free list.
    /// Returns an error if the page ID does not exist.
    pub fn deallocate_page(&mut self, page_id: PageId) -> Result<()> {
        if let Some(offset) = self.pages.remove(&page_id) {
            self.free_slots.push_back(offset);
            Ok(())
        } else {
            Err(Error::InvalidInput(format!(
                "Page ID {} not found",
                page_id
            )))
        }
    }

    /// Read a page if it exists. If not found, returns None or an error.
    pub(crate) fn read(&mut self, page_id: PageId) -> Result<Option<Bytes>> {
        let offset = match self.pages.get(&page_id) {
            Some(&off) => off,
            None => {
                // Not found in pages_, data doesn't exist
                return Ok(None);
            }
        };

        let mut file = self.file.borrow_mut();
        file.seek(SeekFrom::Start(offset))?;

        let mut bytes = BytesMut::zeroed(PAGE_SIZE_BYTES);
        file.read_exact(&mut bytes)?;
        Ok(Some(bytes.freeze()))
    }

    /// Write data to a page. Must not exceed PAGE_SIZE_BYTES.
    pub(crate) fn write(&mut self, page_id: PageId, data: &[u8]) -> Result<()> {
        if data.len() > PAGE_SIZE_BYTES {
            return errdata!("Page data must fit in a page.");
        }

        // If we don't already have an offset for this page, allocate a new one.
        let offset = match self.pages.get(&page_id) {
            Some(&off) => off,
            None => {
                let off = self.allocate_offset()?; // e.g. reuses a free slot or appends
                self.pages.insert(page_id, off);
                off
            }
        };

        let mut file = self.file.borrow_mut();
        file.seek(std::io::SeekFrom::Start(offset))?;
        file.write_all(data)?;
        file.sync_all()?;

        Ok(())
    }

    /// Helper: For new pages, we either reuse a free offset from `free_slots_` or append at the end.
    fn allocate_offset(&mut self) -> Result<u64> {
        // If we have a free offset from a previously deleted page, reuse it
        if let Some(off) = self.free_slots.pop_front() {
            return Ok(off);
        }

        // Otherwise, offset is pages_.len() * PAGE_SIZE_BYTES,
        // but only if we have capacity
        let used_pages = self.pages.len() as u64;
        if used_pages + 1 >= self.page_capacity as u64 {
            // resize (double capacity) if needed
            self.page_capacity *= 2;
            self.resize_file()?;
        }

        // The new offset is used_pages * PAGE_SIZE_BYTES
        let offset = used_pages * PAGE_SIZE_BYTES as u64;
        Ok(offset)
    }

    /// Actually resizes the underlying file to (page_capacity + 1) * PAGE_SIZE_BYTES
    fn resize_file(&mut self) -> Result<()> {
        let size = (self.page_capacity as u64 + 1) * PAGE_SIZE_BYTES as u64;
        let file = self.file.borrow();
        file.set_len(size)
            .map_err(|e| Error::IO(format!("Failed to resize file: {}", e)))?;
        Ok(())
    }

    /// Returns the current size of the database file.
    pub fn get_db_file_size(&self) -> Result<u64> {
        let file = self.file.borrow();
        file.metadata()
            .map(|meta| meta.len())
            .map_err(|e| Error::IO(format!("Failed to get file size: {}", e)))
    }
}

impl Drop for DiskManager {
    /// We unlock the file when the DiskManager is dropped.
    ///
    /// This ensures that while the DiskManager is running, it has exclusive access
    /// to the database file, preventing other processes from modifying it concurrently.
    /// When the DiskManager is dropped, we release the lock so that other processes
    /// (or a new instance of DiskManager) can access the file safely.
    fn drop(&mut self) {
        if let Err(e) = FileExt::unlock(&*self.file.borrow()) {
            panic!("Failed to unlock file: {}", e);
        }
    }
}


