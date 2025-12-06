use std::mem;

use bytemuck::{Pod, Zeroable};

use crate::frame_handle::{PageFrameMutHandle, PageFrameRefHandle};

/// Header for a B+ tree pages containing metadata that both internal page and leaf page share.
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Debug)]
pub(crate) struct BPlusTreePageHeader {
    size: usize,       // Number of values stored in the page.
    max_size: usize,   // Max number of values that can be stored in the page
    page_type: u8,     // Page type (internal, leaf)
    _padding: [u8; 7], // Padding for alignment
}

/// The header occupies the first few bytes of the page.
pub(crate) const BPLUS_TREE_PAGE_HEADER_SIZE: usize = mem::size_of::<BPlusTreePageHeader>();

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// Defines the type of an index page in the B+ tree.
pub(crate) enum BPlusTreePageType {
    InvalidPage = 0, // Neither a leaf nor an internal page (e.g., a meta page or an uninitialized page)
    LeafPage = 1,
    InternalPage = 2,
}

impl BPlusTreePageHeader {
    pub(crate) fn get_page_type_from_frame_ref(handle: &PageFrameRefHandle) -> BPlusTreePageType {
        BPlusTreePageHeader::header_ref(handle).page_type()
    }

    pub(crate) fn get_page_type_from_frame_mut(handle: &PageFrameMutHandle) -> BPlusTreePageType {
        BPlusTreePageHeader::header_mut(handle).page_type()
    }

    pub(crate) fn header_ref<'a>(handle: &'a PageFrameRefHandle<'a>) -> &'a BPlusTreePageHeader {
        let data = handle.data();
        bytemuck::from_bytes::<BPlusTreePageHeader>(&data[..BPLUS_TREE_PAGE_HEADER_SIZE])
    }

    pub(crate) fn header_mut<'a>(handle: &'a PageFrameMutHandle<'a>) -> &'a BPlusTreePageHeader {
        let data = handle.data();
        bytemuck::from_bytes::<BPlusTreePageHeader>(&data[..BPLUS_TREE_PAGE_HEADER_SIZE])
    }

    /// Gets the current size of the page (number of values stored).
    pub(crate) fn size(&self) -> usize {
        self.size
    }

    /// Sets the current size of the page.
    pub(crate) fn set_size(&mut self, size: usize) {
        assert!(
            size <= self.max_size,
            "Current size cannot be greater than the maximum size."
        );
        self.size = size;
    }

    /// Gets the maximum number of values that can be stored.
    pub(crate) fn max_size(&self) -> usize {
        self.max_size
    }

    /// Sets the maximum number of values that can be stored.
    pub(crate) fn set_max_size(&mut self, max_size: usize) {
        self.max_size = max_size;
    }

    /// Gets the page type as `IndexPageType`.
    pub(crate) fn page_type(&self) -> BPlusTreePageType {
        match self.page_type {
            1 => BPlusTreePageType::LeafPage,
            2 => BPlusTreePageType::InternalPage,
            _ => BPlusTreePageType::InvalidPage,
        }
    }

    /// Sets the page type.
    pub(crate) fn set_page_type(&mut self, page_type: BPlusTreePageType) {
        self.page_type = page_type as u8;
    }

    pub(crate) fn min_size(&self) -> usize {
        match self.page_type() {
            BPlusTreePageType::LeafPage => self.max_size() / 2,
            BPlusTreePageType::InternalPage => (self.max_size() + 1) / 2,
            BPlusTreePageType::InvalidPage => {
                panic!("Invoked min size method on a page with invalid type");
            }
        }
    }
}
