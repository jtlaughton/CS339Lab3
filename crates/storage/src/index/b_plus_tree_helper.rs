use super::b_plus_tree::BPlusTreeIndexImpl;

use rustdb_error::{Error, Result};
use std::{
    collections::{HashMap, HashSet},
    fs,
    process::Command,
    sync::{Arc, RwLock},
};

use crate::{
    buffer_pool::BufferPoolManager,
    page::{
        b_plus_tree_internal_page::BPlusTreeInternalPageRef,
        b_plus_tree_leaf_page::BPlusTreeLeafPageRef,
        b_plus_tree_meta_page::BPlusTreeMetaPageRef,
        b_plus_tree_page_header::{BPlusTreePageHeader, BPlusTreePageType},
    },
    typedef::PageId,
};

impl BPlusTreeIndexImpl {
    /// Validates the entire B+ Tree structure and key ordering constraints.
    ///
    /// This is useful for testing or debugging to ensure the tree remains valid after operations.
    ///
    /// What this method checks:
    /// - The tree structure has no cycles (no page has multiple parents).
    /// - Leaf and internal pages are within their allowed size limits.
    /// - Keys in each page are sorted in ascending order.
    /// - Keys respect provided lower and upper bounds based on their parent node.
    ///
    /// This validation starts from the root page and recursively checks all child pages.
    /// It collects visited pages to detect structural problems like multiple parents.
    ///
    /// Call this method after insertions or deletions to confirm your implementation maintains a valid tree.
    ///
    /// Example:
    /// ```
    /// bplus_tree.validate_tree()?;
    /// ```
    pub fn validate_tree(&self) -> Result<()> {
        let bpm = Arc::clone(&self.bpm);
        let meta_page_handle = BufferPoolManager::fetch_page_handle(&bpm, self.meta_page_id)?;
        let meta_page = BPlusTreeMetaPageRef::from(meta_page_handle);
        let mut visited_page_id: HashMap<PageId, PageId> = HashMap::new();
        // Start with no key bounds at the root.
        if let Some(root_page_id) = meta_page.root_page_id() {
            self.validate_node(
                &bpm,
                root_page_id,
                true,
                &mut visited_page_id,
                0,
                None,
                None,
            )?;
        }
        Ok(())
    }

    /// Recursively validates the node with the given page_id.
    /// Additionally, it checks that keys are in sorted order and, if provided,
    /// that they lie within the [lower_bound, upper_bound] range.
    fn validate_node(
        &self,
        bpm: &Arc<RwLock<BufferPoolManager>>,
        page_id: PageId,
        is_root: bool,
        visited: &mut HashMap<PageId, PageId>,
        parent: PageId,
        lower_bound: Option<u32>,
        upper_bound: Option<u32>,
    ) -> Result<()> {
        let page_handle = BufferPoolManager::fetch_page_handle(bpm, page_id)?;
        let page_type = BPlusTreePageHeader::get_page_type_from_frame_ref(&page_handle);
        // Check 1: Detect structural cycles (no page has multiple parents)
        if visited.contains_key(&page_handle.page_id()) {
            panic!(
            "Page with page id {} already visited: a page cannot have more than one parent. First visited from parent with id: {}, now visited again from parent with id: {}",
            page_id,
            visited.get(&page_id).unwrap(),
            parent
        );
        }
        visited.insert(page_id, parent);

        match page_type {
            BPlusTreePageType::LeafPage => {
                let leaf_page = BPlusTreeLeafPageRef::from(page_handle);
                let size = leaf_page.size();

                // Check 2: Validate page size constraints (min and max limits)
                if !is_root {
                    if size < leaf_page.min_size() {
                        return Err(Error::InvalidData(format!(
                            "Leaf page {} has size {} which is less than its minimum {}",
                            page_id,
                            size,
                            leaf_page.min_size()
                        )));
                    }
                    if size > leaf_page.max_size() {
                        return Err(Error::InvalidData(format!(
                            "Leaf page {} has size {} which exceeds its maximum {}",
                            page_id,
                            size,
                            leaf_page.max_size()
                        )));
                    }
                }
                // Validate ordering and bounds of keys in leaf nodes.
                let keys_bytes = leaf_page.key_array();
                // Extract keys as u32 for comparison
                let keys: Vec<u32> = keys_bytes
                    .iter()
                    .map(|&k| u32::from_be_bytes(k))
                    .take(size)
                    .collect();
                if !keys.is_empty() {
                    // Check 3: Validate key order (ascending)
                    for i in 0..keys.len() - 1 {
                        if keys[i] > keys[i + 1] {
                            return Err(Error::InvalidData(format!(
                                "Leaf page {}: keys are not sorted: {} > {}",
                                page_id,
                                keys[i],
                                keys[i + 1]
                            )));
                        }
                    }
                    // Check 4: Validate keys respect parent-provided bounds
                    if let Some(lb) = lower_bound {
                        if keys[0] < lb {
                            return Err(Error::InvalidData(format!(
                                "Leaf page {}: key {} is below lower bound {}",
                                page_id, keys[0], lb
                            )));
                        }
                    }
                    if let Some(ub) = upper_bound {
                        if keys[keys.len() - 1] > ub {
                            return Err(Error::InvalidData(format!(
                                "Leaf page {}: key {} is above upper bound {}",
                                page_id,
                                keys[keys.len() - 1],
                                ub
                            )));
                        }
                    }
                }
            }
            BPlusTreePageType::InternalPage => {
                let internal_page = BPlusTreeInternalPageRef::from(page_handle);
                let size = internal_page.size();
                if !is_root {
                    if size < internal_page.min_size() {
                        return Err(rustdb_error::Error::IO(format!(
                            "Internal page {} has size {} which is less than its minimum {}",
                            page_id,
                            size,
                            internal_page.min_size()
                        )));
                    }
                    if size > internal_page.max_size() {
                        return Err(rustdb_error::Error::IO(format!(
                            "Internal page {} has size {} which exceeds its maximum {}",
                            page_id,
                            size,
                            internal_page.max_size()
                        )));
                    }
                }
                // Validate ordering and bounds of keys in internal nodes.
                let keys_bytes = internal_page.key_array();
                let keys: Vec<u32> = keys_bytes
                    .iter()
                    .map(|&k| u32::from_be_bytes(k))
                    .take(size)
                    .collect();
                if !keys.is_empty() {
                    if let Some(lb) = lower_bound {
                        if keys[1] < lb {
                            return Err(rustdb_error::Error::IO(format!(
                                "Internal page {}: key {} is below lower bound {}",
                                page_id, keys[0], lb
                            )));
                        }
                    }
                    if let Some(ub) = upper_bound {
                        if *keys.last().unwrap() > ub {
                            return Err(rustdb_error::Error::IO(format!(
                                "Internal page {}: key {} is above upper bound {}",
                                page_id,
                                keys[keys.len() - 1],
                                ub
                            )));
                        }
                    }
                    for i in 1..keys.len() - 1 {
                        if keys[i] > keys[i + 1] {
                            return Err(rustdb_error::Error::IO(format!(
                                "Internal page {}: keys are not sorted: {} > {}",
                                page_id,
                                keys[i],
                                keys[i + 1]
                            )));
                        }
                    }
                }
                // Recursively validate each child.
                // For an internal node with N keys and N child pointers,
                // for the first child, the upper bound is keys[0];
                // for each subsequent child i (i ≥ 1), the bounds are [keys[i-1], keys[i]].
                let child_ids = internal_page.pid_array();
                for (i, &child_page_id) in child_ids.iter().take(size).enumerate() {
                    let child_lower = if i == 0 {
                        lower_bound
                    } else {
                        Some(keys[i - 1])
                    };
                    let child_upper = if i == size - 1 {
                        upper_bound
                    } else {
                        Some(keys[i + 1])
                    };
                    self.validate_node(
                        bpm,
                        child_page_id,
                        false,
                        visited,
                        page_id,
                        child_lower,
                        child_upper,
                    )?;
                }
            }
            _ => {
                panic!("Found a page where page type is invalid, either page content is corrupt or you didn't initialize the page");
            }
        }
        Ok(())
    }

    /// Generates a visual PDF diagram of the B+ Tree structure using Mermaid.
    ///
    /// This method converts the tree into Mermaid syntax, saves it as a .mmd file,
    /// and then runs the Mermaid CLI (mmdc) to generate a .pdf diagram.
    ///
    /// Usage instructions:
    /// - You must have the Mermaid CLI installed on your system.
    ///   You can install it globally using:
    ///     npm install -g @mermaid-js/mermaid-cli
    /// - Make sure the `mmdc` command is available in your system PATH.
    ///
    /// Example usage:
    /// ```
    /// bplus_index.generate_visualization(format!("tree after inserting {}", key).as_ref());
    /// ```
    ///
    /// This will create:
    /// - mmd/tree after inserting <key>.mmd  (Mermaid diagram source)
    /// - visualization/tree after inserting <key>.pdf  (Rendered tree visualization)
    ///
    /// # Arguments
    /// - `filename`: The name (without extension) to use for the output files.
    pub fn generate_visualization(&self, filename: &str) -> Result<()> {
        let mermaid_str = self.gen_mermaid_tree_str()?;

        let mmd_dir = "mmd";
        let pdf_dir = "visualization";

        fs::create_dir_all(mmd_dir)?;
        fs::create_dir_all(pdf_dir)?;

        let mmd_filepath = format!("{}/{}.mmd", mmd_dir, filename);
        let pdf_filepath = format!("{}/{}.pdf", pdf_dir, filename);

        // Write Mermaid diagram to a file
        fs::write(&mmd_filepath, mermaid_str)?;

        // Convert `.mmd` file to `.pdf` using Mermaid CLI
        let status = Command::new("mmdc")
            .args(&["-i", &mmd_filepath, "-o", &pdf_filepath])
            .status()
            .expect("Failed to execute Mermaid CLI");

        if !status.success() {
            panic!("failed to generate mermaid chart");
        }

        Ok(())
    }

    fn gen_mermaid_tree_str(&self) -> Result<String> {
        let bpm = Arc::clone(&self.bpm);
        let mut output = String::new();
        output.push_str("flowchart TD\n");

        let meta_page_handle = BufferPoolManager::fetch_page_handle(&bpm, self.meta_page_id)?;
        let meta_page = BPlusTreeMetaPageRef::from(meta_page_handle);
        let meta_node_id = format!("meta_{}", self.meta_page_id);
        output.push_str(&format!(
            "    {}[\"Meta Page root: {:?}\"]\n",
            meta_node_id,
            meta_page.root_page_id()
        ));

        if let Some(root_page_id) = meta_page.root_page_id() {
            let mut visited = HashSet::new();
            self.recursive_print_mermaid(
                &bpm,
                root_page_id,
                &meta_node_id,
                &mut output,
                &mut visited,
            )?;
        }

        Ok(output)
    }

    /// Recursively append the Mermaid.js representation for the page with the given page_id.
    fn recursive_print_mermaid(
        &self,
        bpm: &Arc<RwLock<BufferPoolManager>>,
        page_id: PageId,
        parent_node_id: &str,
        output: &mut String,
        visited: &mut HashSet<PageId>,
    ) -> Result<()> {
        let current_node_id = format!("node_{}", page_id);
        output.push_str(&format!("    {} --> {}\n", parent_node_id, current_node_id));
        if visited.contains(&page_id) {
            return Ok(());
        }
        visited.insert(page_id);

        let page_handle = BufferPoolManager::fetch_page_handle(bpm, page_id)?;
        let page_type = BPlusTreePageHeader::get_page_type_from_frame_ref(&page_handle);

        match page_type {
            BPlusTreePageType::LeafPage => {
                let leaf_page = BPlusTreeLeafPageRef::from(page_handle);
                let keys: Vec<u32> = leaf_page
                    .key_array()
                    .iter()
                    .map(|&key| u32::from_be_bytes(key))
                    .collect();
                let keys = &keys[..leaf_page.size()];
                let values = leaf_page.rid_array().to_vec();
                // Display the values if you want
                let _ = &values[..leaf_page.size()];
                output.push_str(&format!(
                    "{}[\"Leaf (pid: {}) <br> keys: {:?} <br> next: {:?}\"]\n",
                    current_node_id,
                    page_id,
                    keys,
                    leaf_page.next_page_id()
                ));
            }
            _ => {
                // Assume any non-leaf page is an internal page.
                let internal_page = BPlusTreeInternalPageRef::from(page_handle);

                let keys: Vec<u32> = internal_page
                    .key_array()
                    .iter()
                    .map(|&key| u32::from_be_bytes(key))
                    .collect();

                let values: Vec<PageId> = internal_page.pid_array().to_vec();
                let keys = &keys[..internal_page.size()];
                // Display the values if you want
                let _ = &values[..internal_page.size()];
                output.push_str(&format!(
                    "{}[\"Internal (pid: {}) <br> keys: {:?}\"]\n",
                    current_node_id, page_id, keys
                ));
                for &child_page_id in internal_page.pid_array().iter().take(internal_page.size()) {
                    self.recursive_print_mermaid(
                        bpm,
                        child_page_id,
                        &current_node_id,
                        output,
                        visited,
                    )?;
                }
            }
        }

        Ok(())
    }
}
