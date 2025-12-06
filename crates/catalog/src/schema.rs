use crate::column::Column;
use rustdb_error::{Error, Result};
use std::sync::Arc;
/// Can be converted to and from a [`rustdb_storage::record_id::RecordId`] via From/Into trait.
pub type RecordId = u64;
pub type SchemaRef = Arc<Schema>;

/// The schema of a tuple. Contains metadata about the columns corresponding to the tuple's values.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Schema {
    /// The column definitions of the tuple's values, in order.
    columns: Vec<Column>,
    /// The fixed-length size of the tuple, i.e. the sum of the number of bytes used by the fixed
    /// size fields and offsets of the variable length fields. Note specifically that if there are
    /// variable-length fields in the schema, this number excludes their field sizes, but _does_
    /// include the size of their offsets into the data payload.
    size: usize,
}

impl Schema {
    pub fn new(columns: &[Column]) -> Self {
        let columns = columns.to_vec();
        let size = columns.iter().filter_map(|c| c.size()).sum();
        Schema { columns, size }
    }

    /// Moves all the columns of `other` into `self`, consuming `other`.
    ///
    /// Because `Schema` stores its columns in a `Vec`, this method will panic if the new column
    /// vector capacity exceeds `isize::MAX` _bytes_ (see [`Vec::append`]).
    pub fn append(&mut self, mut other: Self) {
        self.size += other.size;
        self.columns.append(&mut other.columns);
    }

    /// Returns an immutable view of the columns.
    pub fn columns(&self) -> &Vec<Column> {
        &self.columns
    }

    /// Given an index `i`, retrieves a reference to the schema's `i`-th column (if it exists).
    pub fn column_at(&self, index: usize) -> Result<&Column> {
        self.columns.get(index).ok_or(Error::OutOfBounds)
    }

    /// Looks and up and returns the index of the column in the schema with the given name, if
    /// one exists. If more than one column has the given name, returns the index of the first
    /// such column.
    pub fn column_index_of(&self, name: &str) -> Option<usize> {
        self.columns
            .iter()
            .enumerate()
            .find(|(_i, col)| col.name() == name)
            .map(|(i, _)| i)
    }

    /// Returns the number of columns in the schema.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Returns the fixed-length byte size of the tuple.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl std::fmt::Display for Schema {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let columns = self
            .columns
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",");
        write!(
            f,
            "Schema[ NumColumns: {}, FixedSize: {} ] :: ( {} )",
            self.num_columns(),
            self.size(),
            columns
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::schema::Schema;
    use crate::types::Type;
    use rand::Rng;
    use rustdb_error::Error;

    #[test]
    fn test_append() {
        let all_columns = create_n_columns(10);

        // Create a schema with the first half of the columns and another schema with the rest.
        let (mut first, second) = {
            let (first, second) = all_columns.split_at(5);
            (Schema::new(first), Schema::new(second))
        };
        let total_size = first.size() + second.size();

        // Appending the two schemas together should mutate the first into a larger schema that
        // contains all the columns.
        first.append(second);
        assert_eq!(first.size(), total_size);
        assert_eq!(first.columns(), &all_columns);
    }

    #[test]
    fn test_equality() {
        let names = vec!["a", "b", "c"];
        let integer_columns = names
            .iter()
            .map(|name| Column::new(name.to_string(), Type::Integer))
            .collect::<Vec<_>>();
        let float_columns = names
            .iter()
            .map(|name| Column::new(name.to_string(), Type::Float))
            .collect::<Vec<_>>();

        // Schemas with matching column names BUT with differing column types aren't equal.
        assert_ne!(Schema::new(&integer_columns), Schema::new(&float_columns));

        // Schemas with matching column vectors (even if they're different objects) are equal.
        let more_integer_columns = integer_columns.clone();
        assert_eq!(Schema::new(&integer_columns), Schema::new(&integer_columns));
    }

    #[test]
    fn test_num_columns() {
        let random = {
            let mut rng = rand::rng();
            rng.random_range(1..100)
        };
        assert_eq!(0, Schema::new(&create_n_columns(0)).num_columns());
        assert_eq!(random, Schema::new(&create_n_columns(random)).num_columns());
    }

    #[test]
    fn test_column_at() {
        // This is a really wide row.
        let columns = create_n_columns(50);
        let schema = Schema::new(&columns);

        // Retrieving by valid column indexes returns the corresponding column reference.
        (0..50).for_each(|i| {
            assert_eq!(schema.column_at(i), Ok(&columns[i]));
        });

        // Trying to access an out-of-bounds index results in an error.
        assert_eq!(schema.column_at(50).err(), Some(Error::OutOfBounds));
    }

    #[test]
    fn test_column_index_of() {
        // Via `create_n_columns()`, a column's index is also its name, e.g. the name of the column
        // at index i=3 is "3". We'll add another column (with a duplicate name!) as well, so that
        // `columns` is a vector of 11 columns with corresponding names {"0", "1", ... , "9", "0"}.
        let columns = {
            let mut columns = create_n_columns(10);
            columns.push(Column::new("0".to_string(), Type::Integer));
            columns
        };
        let schema = Schema::new(&columns);

        // In the case of a duplicate name, we get the index of the first such column with it.
        assert_eq!(schema.column_index_of("0"), Some(0));

        // We should be able to retrieve the indexes of all other valid column names.
        (1..10).for_each(|i| {
            let name = i.to_string();
            assert_eq!(schema.column_index_of(&name), Some(i))
        });

        // If none of the columns have the requested name, we should get an empty option.
        assert!(schema.column_index_of("10").is_none());
        assert!(schema.column_index_of("").is_none());
        assert!(schema.column_index_of("All love 🛸💕🕺").is_none());
    }

    fn create_n_columns(n: usize) -> Vec<Column> {
        (0..n)
            .map(|i| Column::new(i.to_string(), Type::Null))
            .collect::<Vec<_>>()
    }
}
