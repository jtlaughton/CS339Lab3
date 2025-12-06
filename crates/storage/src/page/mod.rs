use crate::typedef::PageId;

pub(crate) mod b_plus_tree_internal_page;
pub(crate) mod b_plus_tree_leaf_page;
pub(crate) mod b_plus_tree_meta_page;
pub(crate) mod b_plus_tree_page_header;
pub(crate) mod table_page;

pub(crate) const INVALID_PAGE_ID: PageId = 0;
pub(crate) const PAGE_SIZE: usize = 4096;
