use std::hash::Hash;
use std::sync::Arc;

/// A map of identifiers to async read-write locks.
pub(crate) struct LockManager<T, I>
where
    T: Eq + PartialEq + Hash,
{
    locks: std::sync::Mutex<std::collections::HashMap<T, Arc<tokio::sync::RwLock<Option<I>>>>>,
}

impl<T, I> LockManager<T, I> where T: Eq + PartialEq + Hash {}
