use crate::{page::INVALID_PAGE_ID, typedef::PageId};

#[derive(Clone, Debug, Hash)]
pub struct RecordId {
    /// The ID of the page the record lives inside.
    page_id: PageId,
    /// The offset of the record in the list of page tuples. Not to be confused with the byte
    /// offset of the tuple in the page data!
    slot_id: u32,
}

pub const INVALID_RECORD_ID: RecordId = RecordId {
    page_id: INVALID_PAGE_ID,
    slot_id: 0,
};

/// The specific fields of a `RecordId` won't be of importance in every context that a record
/// id will be passed around; sometimes, all we need is an identifier of the record itself
/// (and not the extra information embedded into its representation).
///
/// In those cases, it can be more convenient to pass around the record id as an int than the
/// struct in full. This provides an easy way to convert between these two representations.
impl From<u64> for RecordId {
    fn from(value: u64) -> Self {
        Self {
            page_id: (value >> 32) as PageId,
            slot_id: value as u32,
        }
    }
}
impl From<RecordId> for u64 {
    fn from(record: RecordId) -> Self {
        u64::from(record.page_id) << 32 | u64::from(record.slot_id)
    }
}

impl RecordId {
    pub fn new(page_id: PageId, sid: u32) -> RecordId {
        RecordId {
            page_id,
            slot_id: sid,
        }
    }

    pub fn to_string(&self) -> String {
        format!("{}:{}", self.page_id, self.slot_id)
    }

    pub fn page_id(&self) -> PageId {
        self.page_id
    }

    pub fn slot_id(&self) -> u32 {
        self.slot_id
    }
}

impl PartialEq<Self> for RecordId {
    fn eq(&self, other: &Self) -> bool {
        self.page_id == other.page_id && self.slot_id == other.slot_id
    }
}

impl Eq for RecordId {} // implement Eq trait for RecordId, uses PartialEq

impl Ord for RecordId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.page_id == other.page_id {
            if self.slot_id < other.slot_id {
                return std::cmp::Ordering::Less;
            } else if self.slot_id > other.slot_id {
                return std::cmp::Ordering::Greater;
            } else {
                return std::cmp::Ordering::Equal;
            }
        } else if self.page_id < other.page_id {
            return std::cmp::Ordering::Less;
        } else {
            return std::cmp::Ordering::Greater;
        }
    }
}

impl PartialOrd for RecordId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}


