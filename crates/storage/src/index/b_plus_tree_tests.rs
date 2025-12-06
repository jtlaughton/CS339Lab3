#[cfg(test)]
mod tests {
    use rustdb_catalog::schema::RecordId;
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex, RwLock},
    };

    use crate::index::b_plus_tree::{self, BPlusTreeIndex, BPlusTreeIndexImpl};

    use crate::{
        buffer_pool::BufferPoolManager, disk::disk_manager::DiskManager,
        index::b_plus_tree::BPlusTreeKey, replacer::lru_k_replacer::LrukReplacer,
    };
    use rand::Rng;
    use rand::{
        rng,
        seq::{IteratorRandom, SliceRandom},
    };
    use rustdb_error::Result;
    use serial_test::serial;

    /// Generates a BPlusTreeKey with sequential values starting from `x`.
    fn get_key_from_u32(x: u32) -> BPlusTreeKey {
        x.to_be_bytes()
    }

    /// Generates a random RecordId.
    fn gen_rid() -> RecordId {
        rand::random::<RecordId>() % 1000
    }

    // Create a BufferPoolManager with the given pool size.
    fn get_bpm_with_pool_size(pool_size: usize) -> BufferPoolManager {
        let disk_manager = Arc::new(Mutex::new(DiskManager::new("test.db").unwrap()));
        let replacer = Box::new(LrukReplacer::new(5));
        BufferPoolManager::new(pool_size, disk_manager, replacer)
    }

    fn get_bpm_arc_with_pool_size(pool_size: usize) -> Arc<RwLock<BufferPoolManager>> {
        Arc::new(RwLock::new(get_bpm_with_pool_size(pool_size)))
    }

    #[test]
    #[serial]
    fn test_index_b_plus_tree_get() -> Result<()> {
        let pool_size = 20;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        let bplus_index = BPlusTreeIndexImpl::new("test_index".to_string(), Arc::clone(&bpm), 3, 3);

        for i in 0..300 {
            let k: BPlusTreeKey = get_key_from_u32(i);
            let v: RecordId = (i as u32).into();
            assert!(bplus_index.insert(&k, v).is_ok());
            assert_eq!(bplus_index.get(&k), Some(v));

            let handle = bpm.write()?;
            // make sure all frame handles are dropped and unpinned
            assert_eq!(handle.free_frame_count(), pool_size);
        }

        let handle = bpm.write().unwrap();
        // make sure all frame handles are dropped and unpinned
        assert_eq!(handle.free_frame_count(), pool_size);

        Ok(())
    }

    #[test]
    #[serial]
    fn test_index_b_plus_tree_random_insert_and_get() -> Result<()> {
        let pool_size = 20;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        let bplus_index = BPlusTreeIndexImpl::new("test_index".to_string(), Arc::clone(&bpm), 3, 3);

        let mut keys: Vec<u32> = (250..1000).collect();
        let mut rng = rand::rng();
        keys.shuffle(&mut rng);

        for &i in &keys {
            let k: BPlusTreeKey = get_key_from_u32(i);
            let v: RecordId = i.into();
            assert!(bplus_index.insert(&k, v).is_ok());
            assert_eq!(bplus_index.get(&k), Some(v));
            let handle = bpm.write().unwrap();
            assert_eq!(handle.free_frame_count(), pool_size);
        }

        for i in 0..249 {
            let missing_key: BPlusTreeKey = get_key_from_u32(i);
            assert_eq!(bplus_index.get(&missing_key), None);
        }

        Ok(())
    }

    #[test]
    #[serial]
    fn test_index_b_plus_tree_random_ops_insert_get() -> Result<()> {
        let pool_size = 50;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        let index = BPlusTreeIndexImpl::new("random_test".to_string(), Arc::clone(&bpm), 5, 5);

        let mut reference: HashMap<BPlusTreeKey, RecordId> = HashMap::new();

        let mut rng = rand::rng();

        let num_ops = 10000;

        for _ in 0..num_ops {
            let key_val = rng.random_range(0..1000);
            let key = get_key_from_u32(key_val);

            let do_insert = rng.random_bool(0.5);

            if do_insert {
                let value: RecordId = rng.random_range(0..1000);
                if reference.get(&key).is_some() {
                    let tree_result = index.get(&key);
                    assert!(tree_result.is_some());
                    continue;
                }
                index.insert(&key, value)?;
                reference.insert(key, value);
            } else {
                let tree_result = index.get(&key);
                let ref_result = reference.get(&key).copied();
                assert_eq!(tree_result, ref_result);
            }
        }

        Ok(())
    }

    #[test]
    #[serial]
    fn test_index_b_plus_tree_remove_easy() -> Result<()> {
        let pool_size = 50;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        let bplus_index = BPlusTreeIndexImpl::new("test_index".to_string(), Arc::clone(&bpm), 3, 3);

        for i in 0..50 {
            let k: BPlusTreeKey = get_key_from_u32(i);
            let v: RecordId = (i as u32).into();
            assert!(bplus_index.insert(&k, v).is_ok());
            assert_eq!(bplus_index.get(&k), Some(v));

            let handle = bpm.write().unwrap();
            // make sure all frame handles are dropped and unpinned
            assert_eq!(handle.free_frame_count(), pool_size);
        }

        for i in 15..16 {
            let k: BPlusTreeKey = get_key_from_u32(i);
            assert!(bplus_index.remove(&k).is_some());
            assert_eq!(bplus_index.get(&k), None);

            let handle = bpm.write().unwrap();
            // make sure all frame handles are dropped and unpinned
            assert_eq!(handle.free_frame_count(), pool_size);
        }

        let handle = bpm.write().unwrap();
        // make sure all frame handles are dropped and unpinned
        assert_eq!(handle.free_frame_count(), pool_size);

        Ok(())
    }

    #[test]
    #[serial]
    fn test_index_b_plus_tree_remove_medium() -> Result<()> {
        let pool_size = 50;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        let bplus_index = BPlusTreeIndexImpl::new("test_index".to_string(), Arc::clone(&bpm), 3, 3);

        for i in 0..50 {
            let k: BPlusTreeKey = get_key_from_u32(i);
            let v: RecordId = (i as u32).into();
            assert!(bplus_index.insert(&k, v).is_ok());
            assert_eq!(bplus_index.get(&k), Some(v));

            let handle = bpm.write().unwrap();
            // make sure all frame handles are dropped and unpinned
            assert_eq!(handle.free_frame_count(), pool_size);
        }

        for i in 15..40 {
            let k: BPlusTreeKey = get_key_from_u32(i);
            assert!(bplus_index.remove(&k).is_some());
            assert_eq!(bplus_index.get(&k), None);

            let handle = bpm.write().unwrap();
            // make sure all frame handles are dropped and unpinned
            assert_eq!(handle.free_frame_count(), pool_size);
        }

        let handle = bpm.write().unwrap();
        // make sure all frame handles are dropped and unpinned
        assert_eq!(handle.free_frame_count(), pool_size);

        Ok(())
    }

    #[test]
    #[serial]
    fn test_index_b_plus_tree_remove() -> Result<()> {
        let pool_size = 20;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        let bplus_index = BPlusTreeIndexImpl::new("test_index".to_string(), Arc::clone(&bpm), 3, 4);

        for i in 0..300 {
            let k: BPlusTreeKey = get_key_from_u32(i);
            let v: RecordId = i.into();
            assert!(bplus_index.insert(&k, v).is_ok());
            assert_eq!(bplus_index.get(&k), Some(v));
        }

        for i in 50..250 {
            let k: BPlusTreeKey = get_key_from_u32(i);
            let res = bplus_index.remove(&k);
            assert!(res.is_some());
            assert_eq!(bplus_index.get(&k), None);
            bplus_index.validate_tree()?;
        }

        for i in 50..250 {
            let k: BPlusTreeKey = get_key_from_u32(i);
            let v: RecordId = i.into();
            assert!(bplus_index.insert(&k, v).is_ok());
            bplus_index.validate_tree()?;
        }

        for i in 0..300 {
            let k: BPlusTreeKey = get_key_from_u32(i);
            let v: RecordId = i.into();
            bplus_index.validate_tree()?;
            assert_eq!(bplus_index.get(&k), Some(v));
        }

        Ok(())
    }

    #[test]
    #[serial]
    fn test_index_b_plus_tree_random_ops_remove_insert_get() -> Result<()> {
        let pool_size = 200;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        let index = BPlusTreeIndexImpl::new("test".to_string(), Arc::clone(&bpm), 3, 2);

        let mut reference: HashMap<BPlusTreeKey, RecordId> = HashMap::new();
        let mut rng = rand::rng();
        let num_ops = 10000;

        for _ in 0..num_ops {
            let key_val = rng.random_range(0..1000);
            let key = get_key_from_u32(key_val);

            let operation = rng.random_range(0..3);

            match operation {
                0 => {
                    let value: RecordId = key_val.into();
                    if let Some(ref_result) = reference.get(&key) {
                        let tree_result = index.get(&key);
                        assert_eq!(tree_result, Some(*ref_result));
                        continue;
                    }
                    index.insert(&key, value)?;
                    reference.insert(key, value);
                }
                1 => {
                    let tree_result = index.get(&key);
                    let ref_result = reference.get(&key).copied();
                    assert_eq!(tree_result, ref_result);
                }
                2 => {
                    let exist = rng.random_range(0..2);
                    if exist == 0 || reference.len() == 0 {
                        let tree_result = index.remove(&key);
                        let ref_result = reference.remove(&key);
                        assert_eq!(tree_result, ref_result);
                    } else {
                        let random_key = reference.keys().choose(&mut rand::rng()).unwrap();
                        let tree_result = index.remove(&random_key.clone());
                        let ref_result = reference.remove(&random_key.clone());
                        assert_eq!(tree_result, ref_result);
                    }
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    #[test]
    #[serial]
    fn test_index_b_plus_easy_tree_split() -> Result<()> {
        let pool_size = 5;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        let bplus_index = BPlusTreeIndexImpl::new("foo_pk".to_string(), Arc::clone(&bpm), 6, 10);

        let keys: Vec<i64> = vec![1, 2, 3, 4, 5];

        for &key in &keys {
            let index_key = (key as u32).to_be_bytes();
            let rid = key as u64;

            assert_eq!(bplus_index.insert(&index_key, rid).is_ok(), true);
            bplus_index.validate_tree()?;
        }

        for &key in &keys {
            let index_key = (key as u32).to_be_bytes();
            let rid = key as u64;

            assert!(bplus_index.insert(&index_key, rid).is_err());
            bplus_index.validate_tree()?;
        }

        bplus_index.validate_tree()?;

        Ok(())
    }

    #[test]
    #[serial]
    fn test_index_b_plus_tree_easy_insert_1() -> Result<()> {
        let pool_size = 5;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        let bplus_index = BPlusTreeIndexImpl::new("foo_pk".to_string(), Arc::clone(&bpm), 6, 10);

        let keys: Vec<i64> = vec![1, 2, 3, 4, 5];

        for &key in &keys {
            let index_key = (key as u32).to_be_bytes();
            let rid = key as u64;
            assert!(bplus_index.insert(&index_key, rid).is_ok());
        }

        for &key in &keys {
            let index_key = (key as u32).to_be_bytes();
            let result = bplus_index.get(&index_key);
            assert!(result.is_some());
            let rid = result.unwrap();
            assert_eq!(rid, key as u64);
        }

        bplus_index.validate_tree()?;
        Ok(())
    }

    #[test]
    #[serial]
    fn test_index_b_plus_easy_tree_merge() -> Result<()> {
        let pool_size = 10;
        let bpm = get_bpm_arc_with_pool_size(pool_size);

        let bplus_index = BPlusTreeIndexImpl::new("foo_pk".to_string(), Arc::clone(&bpm), 3, 2);

        let keys: Vec<i64> = vec![2, 1, 3, 5, 4];

        // Insert keys
        for &key in &keys {
            let index_key = (key as u32).to_be_bytes();
            let rid = key as u64;

            assert_eq!(bplus_index.insert(&index_key, rid).is_ok(), true);
            bplus_index.validate_tree()?;
        }

        let remove_keys: Vec<i64> = vec![1, 5];

        // Remove keys
        for &key in &remove_keys {
            let index_key = (key as u32).to_be_bytes();
            bplus_index.remove(&index_key);
            bplus_index.validate_tree()?;
        }

        // Now find the leaf node containing key 2
        let index_key = (2 as u32).to_be_bytes();

        let result = bplus_index.get(&index_key);
        assert!(result == Some(2));

        Ok(())
    }

    #[test]
    #[serial]
    fn test_index_b_plus_tree_insert_and_sorted_scan() -> Result<()> {
        let pool_size = 5;
        let bpm = get_bpm_arc_with_pool_size(pool_size);
        let bplus_index = BPlusTreeIndexImpl::new("foo_pk".to_string(), Arc::clone(&bpm), 3, 3);

        // Insert keys in reverse order
        let keys: Vec<i64> = vec![5, 4, 3, 2, 1];
        for &key in &keys {
            let index_key = (key as u32).to_be_bytes();
            let rid = key as u64;
            assert!(bplus_index.insert(&index_key, rid).is_ok());
        }

        // Step 1: collect all key-values from test data
        let mut sorted_keys = keys.clone();
        sorted_keys.sort();

        // Scan from key = 1
        let mut current_key = 1;
        for &key in &sorted_keys {
            let index_key = (key as u32).to_be_bytes();
            let result = bplus_index.get(&index_key);
            assert!(result.is_some());
            assert_eq!(result.unwrap(), key as u64);
            assert_eq!(current_key, key);
            current_key += 1;
        }
        assert_eq!(current_key, 6); // keys 1 through 5

        // Scan from key = 3
        let mut current_key = 3;
        for &key in &sorted_keys {
            if key < current_key {
                continue;
            }
            let index_key = (key as u32).to_be_bytes();
            let result = bplus_index.get(&index_key);
            assert!(result.is_some());
            assert_eq!(result.unwrap(), key as u64);
            assert_eq!(current_key, key);
            current_key += 1;
        }
        assert_eq!(current_key, 6); // keys 3 through 5

        Ok(())
    }

    #[test]
    #[serial]
    fn test_index_b_plus_tree_insert_and_delete_scan() -> Result<()> {
        let pool_size = 5;
        let bpm = get_bpm_arc_with_pool_size(pool_size);
        let bplus_index = BPlusTreeIndexImpl::new("foo_pk".to_string(), Arc::clone(&bpm), 3, 3);

        // Insert keys
        let keys: Vec<i64> = vec![1, 2, 3, 4, 5];
        for &key in &keys {
            let index_key = (key as u32).to_be_bytes();
            let rid = key as u64;
            assert!(bplus_index.insert(&index_key, rid).is_ok());
        }

        // Check that all keys can be retrieved correctly
        for &key in &keys {
            let index_key = (key as u32).to_be_bytes();
            let result = bplus_index.get(&index_key);
            assert!(result.is_some());
            assert_eq!(result.unwrap(), key as u64);
        }

        // Manually simulate a forward scan from key = 1
        let mut sorted_keys = keys.clone();
        sorted_keys.sort();
        let mut current_key = 1;
        for &key in &sorted_keys {
            let index_key = (key as u32).to_be_bytes();
            let result = bplus_index.get(&index_key);
            assert!(result.is_some());
            assert_eq!(result.unwrap(), current_key as u64);
            current_key += 1;
        }
        assert_eq!(current_key, 6); // scanned 5 keys

        // Now delete keys 1 and 5
        for &key in &[1, 5] {
            let index_key = (key as u32).to_be_bytes();
            let removed = bplus_index.remove(&index_key);
            assert_eq!(removed, Some(key as u64));
        }

        // Scan from key = 2 and collect valid entries
        let mut remaining = Vec::new();
        for &key in &[2, 3, 4, 5] {
            let index_key = (key as u32).to_be_bytes();
            if let Some(val) = bplus_index.get(&index_key) {
                remaining.push((key, val));
            }
        }

        assert_eq!(remaining.len(), 3);
        assert_eq!(remaining[0], (2, 2));
        assert_eq!(remaining[1], (3, 3));
        assert_eq!(remaining[2], (4, 4));

        Ok(())
    }
    #[test]
    #[serial]
    fn test_index_b_plus_tree_deletion_complex() -> Result<()> {
        let pool_size = 5;
        let bpm = get_bpm_arc_with_pool_size(pool_size);
        let bplus_index = BPlusTreeIndexImpl::new("foo_pk".to_string(), Arc::clone(&bpm), 3, 3);

        // Insert keys: 1, 2, 3, 4, 5
        let keys: Vec<i64> = vec![1, 2, 3, 4, 5];
        for &key in &keys {
            let index_key = (key as u32).to_be_bytes();
            let rid = key as u64;
            assert!(bplus_index.insert(&index_key, rid).is_ok());
        }

        // Validate all keys exist
        for &key in &keys {
            let index_key = (key as u32).to_be_bytes();
            let result = bplus_index.get(&index_key);
            assert!(result.is_some());
            assert_eq!(result.unwrap(), key as u64);
        }

        // Scan all keys in order from key = 1
        let mut sorted_keys = keys.clone();
        sorted_keys.sort();
        let mut current_key = 1;
        for &key in &sorted_keys {
            let index_key = (key as u32).to_be_bytes();
            let result = bplus_index.get(&index_key);
            assert!(result.is_some());
            assert_eq!(result.unwrap(), current_key as u64);
            current_key += 1;
        }
        assert_eq!(current_key, 6); // Keys 1 through 5 were scanned

        // Delete keys: 1, 5, 3, 4
        for &key in &[1, 5, 3, 4] {
            let index_key = (key as u32).to_be_bytes();
            let removed = bplus_index.remove(&index_key);
            assert_eq!(removed, Some(key as u64));
        }

        // Scan from key = 2
        let start_key = 2;
        let mut current_key = start_key;
        let mut size = 0;

        for &key in &sorted_keys {
            if key < start_key {
                continue;
            }

            let index_key = (key as u32).to_be_bytes();
            if let Some(val) = bplus_index.get(&index_key) {
                assert_eq!(val, current_key as u64);
                current_key += 1;
                size += 1;
            }
        }

        assert_eq!(size, 1); // Only key=2 should remain

        Ok(())
    }

    #[test]
    #[serial]
    fn test_index_b_plus_tree_scale() -> Result<()> {
        let pool_size = 20;
        let bpm = get_bpm_arc_with_pool_size(pool_size);
        let bplus_index =
            BPlusTreeIndexImpl::new("scale_index".to_string(), Arc::clone(&bpm), 6, 10);

        let scale = 10_000;
        let mut keys: Vec<i64> = (1..scale).collect();
        keys.shuffle(&mut rng());

        // Insert all keys
        for &key in &keys {
            let k = (key as u32).to_be_bytes();
            let rid = key as u64;
            bplus_index.insert(&k, rid)?;
        }

        // Validate all inserted keys
        for &key in &keys {
            let k = (key as u32).to_be_bytes();
            let result = bplus_index.get(&k);
            assert_eq!(result, Some(key as u64));
        }

        // Scan from key = 1
        let mut current_key = 1;
        let mut sorted_keys: Vec<i64> = (1..scale).collect();
        sorted_keys.sort();
        for &key in &sorted_keys {
            let k = (key as u32).to_be_bytes();
            if let Some(rid) = bplus_index.get(&k) {
                assert_eq!(rid, key as u64);
                current_key += 1;
            }
        }
        assert_eq!(current_key, scale);

        // Remove keys from 1 to 9899
        for key in 1..9900 {
            let k = (key as u32).to_be_bytes();
            bplus_index.remove(&k);
        }

        // Confirm 100 keys remain
        let mut remaining = 0;
        for key in 9900..scale {
            let k = (key as u32).to_be_bytes();
            if bplus_index.get(&k).is_some() {
                remaining += 1;
            }
        }
        assert_eq!(remaining, 100);

        // Remove the last 100
        for key in 9900..scale {
            let k = (key as u32).to_be_bytes();
            bplus_index.remove(&k);
        }

        Ok(())
    }

    #[test]
    #[serial]
    fn test_concurrent_inserts_mixed_order() -> Result<()> {
        use std::collections::HashSet;
        use std::sync::{Arc, Barrier, Mutex};
        use std::thread;

        let pool_size = 50;
        let bpm = get_bpm_arc_with_pool_size(pool_size);
        let bplus_index = Arc::new(BPlusTreeIndexImpl::new(
            "concurrent_test".to_string(),
            Arc::clone(&bpm),
            3,
            3,
        ));

        let barrier = Arc::new(Barrier::new(4));
        let inserted_keys = Arc::new(Mutex::new(HashSet::new()));
        let mut handles = vec![];

        let total_keys = 300;

        let num_threads: u64 = 4;
        for thread_id in 0..4 {
            let index_clone = Arc::clone(&bplus_index);
            let barrier_clone = Arc::clone(&barrier);
            let keys_set = Arc::clone(&inserted_keys);

            let handle = thread::spawn(move || {
                barrier_clone.wait(); // synchronize start

                let range: Box<dyn Iterator<Item = u64>> = match thread_id {
                    0 | 1 => {
                        Box::new((thread_id..total_keys as u64 + 1).step_by(num_threads as usize))
                    }
                    2 | 3 => Box::new(
                        (thread_id
                            ..(total_keys + (total_keys % num_threads) - num_threads + thread_id)
                                as u64
                                + 1)
                            .rev()
                            .step_by(num_threads as usize),
                    ),
                    _ => unreachable!(),
                };

                for key_val in range {
                    let key = get_key_from_u32(key_val as u32);
                    let value: RecordId = key_val.into();
                    index_clone.insert(&key, value).unwrap();
                    keys_set.lock().unwrap().insert(key_val);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let keys = inserted_keys.lock().unwrap().clone();
        for key_val in keys {
            let key = get_key_from_u32(key_val as u32);
            let expected_val: RecordId = key_val.into();
            assert_eq!(bplus_index.get(&key), Some(expected_val));
        }
        Ok(())
    }

    #[test]
    #[serial]
    fn test_concurrent_deletes() -> Result<()> {
        use std::sync::{Arc, Barrier};
        use std::thread;

        let pool_size = 50;
        let bpm = get_bpm_arc_with_pool_size(pool_size);
        let bplus_index = Arc::new(BPlusTreeIndexImpl::new(
            "delete_test".to_string(),
            Arc::clone(&bpm),
            3,
            3,
        ));

        let total_keys: usize = 1000;
        // Step 1: Insert all keys sequentially
        for key_val in 0..=total_keys {
            let key = get_key_from_u32(key_val as u32);
            let value: RecordId = key_val as u64;
            bplus_index.insert(&key, value)?;
        }

        // Step 2: Spawn threads to delete keys concurrently
        let num_threads = 4;
        let barrier = Arc::new(Barrier::new(num_threads));
        let mut handles = vec![];

        for thread_id in 0..num_threads {
            let index_clone = Arc::clone(&bplus_index);
            let barrier_clone = Arc::clone(&barrier);

            let handle = thread::spawn(move || {
                barrier_clone.wait(); // synchronize thread start
                for key_val in (thread_id..=total_keys).step_by(num_threads) {
                    let key = get_key_from_u32(key_val as u32);
                    index_clone.remove(&key).unwrap();
                }
            });

            handles.push(handle);
        }

        // Wait for all deletion threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Step 3: Check that all keys have been deleted
        for key_val in 0..=total_keys {
            let key = get_key_from_u32(key_val as u32);
            assert_eq!(bplus_index.get(&key), None);
        }

        Ok(())
    }

    #[test]
    #[serial]
    fn test_b_plus_tree_iterator_forward_scan() -> Result<()> {
        let pool_size = 5;
        let bpm = get_bpm_arc_with_pool_size(pool_size);
        let bplus_index = BPlusTreeIndexImpl::new("iter_test".to_string(), Arc::clone(&bpm), 3, 3);

        // Insert keys: 1, 2, 3, 4, 5
        let keys: Vec<i64> = vec![1, 2, 3, 4, 5];
        for &key in &keys {
            let index_key = (key as u32).to_be_bytes();
            let rid = key as u64;
            assert!(bplus_index.insert(&index_key, rid).is_ok());
        }

        // Start iteration from key = 3
        let start_key = (3 as u32).to_be_bytes();
        let mut iter = bplus_index.range(Some(&start_key), None);

        let expected_keys = vec![3, 4, 5];
        for &expected in &expected_keys {
            let next = iter.next();
            assert!(next.is_some(), "Expected key {} not found", expected);
            let rid = next.unwrap();
            assert_eq!(rid.1, expected);
        }

        // Ensure no more entries
        assert!(iter.next().is_none());

        Ok(())
    }

    #[test]
    #[serial]
    fn test_b_plus_tree_iterator_forward_scan_large() -> Result<()> {
        let pool_size = 10;
        let bpm = get_bpm_arc_with_pool_size(pool_size);
        let bplus_index =
            BPlusTreeIndexImpl::new("iter_test_large".to_string(), Arc::clone(&bpm), 3, 3);

        // Insert keys: 1 through 20
        let keys: Vec<i64> = (1..=20).collect();
        for &key in &keys {
            let index_key = (key as u32).to_be_bytes();
            let rid = key as u64;
            assert!(bplus_index.insert(&index_key, rid).is_ok());
        }

        // Collect results from iterator starting at key = 10
        let start_key = (10 as u32).to_be_bytes();
        let iter = bplus_index.range(Some(&start_key), None);

        let results: Vec<_> = iter.map(|(_, rid)| rid).collect();

        // Expected values are 10 through 20
        let expected: Vec<_> = (10..=20).map(|n| n as u64).collect();
        assert_eq!(results, expected);

        bplus_index.generate_visualization("iterator")?;

        Ok(())
    }

    #[test]
    #[serial]
    fn test_b_plus_tree_iterator_range_scan_large() -> Result<()> {
        let pool_size = 10;
        let bpm = get_bpm_arc_with_pool_size(pool_size);
        let bplus_index =
            BPlusTreeIndexImpl::new("iter_test_large".to_string(), Arc::clone(&bpm), 3, 3);

        // Insert keys: 1 through 20
        let keys: Vec<i64> = (1..=20).collect();
        for &key in &keys {
            let index_key = (key as u32).to_be_bytes();
            let rid = key as u64;
            assert!(bplus_index.insert(&index_key, rid).is_ok());
        }

        // Collect results from iterator starting at key = 10
        let start_key = (10 as u32).to_be_bytes();
        let end_key = (17 as u32).to_be_bytes();
        let iter = bplus_index.range(Some(&start_key), Some(&end_key));

        let results: Vec<_> = iter.map(|(_, rid)| rid).collect();

        // Expected values are 10 through 17
        let expected: Vec<_> = (10..=17).map(|n| n as u64).collect();
        assert_eq!(results, expected);

        bplus_index.generate_visualization("iterator")?;

        Ok(())
    }
}
