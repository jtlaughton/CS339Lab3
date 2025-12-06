use crate::types::Type;
use std::fmt::Debug;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Column {
    /// The name of this column.
    name: String,
    /// This column value's type.
    field_type: Type,
}

impl Column {
    pub fn new(name: String, field_type: Type) -> Self {
        Column { name, field_type }
    }

    /// Returns the name of this column.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the `field_type` of the column.
    pub fn field_type(&self) -> Type {
        self.field_type
    }

    /// Returns the fixed byte size of this column's field data. In the case of variable-length
    /// fields, returns `None`.
    pub fn size(&self) -> Option<usize> {
        match self.field_type {
            Type::Varchar => None,
            fixed_size_type @ _ => Some(fixed_size_type.size()),
        }
    }
}

impl std::fmt::Display for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let length = match self.field_type {
            Type::Varchar => "VARIABLE".to_string(),
            fixed_size_field @ _ => fixed_size_field.size().to_string(),
        };
        write!(
            f,
            "Column[ {}, {}, Length: {} bytes ]",
            self.name, self.field_type, length
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::types::Type;

    #[test]
    fn test_column_size() {
        // Fixed-length field sizes are as expected:
        assert_eq!(with_type(Type::Null).size(), Some(0));
        assert_eq!(with_type(Type::Boolean).size(), Some(1));
        assert_eq!(with_type(Type::Integer).size(), Some(size_of::<i32>()));
        assert_eq!(with_type(Type::Float).size(), Some(size_of::<f64>()));

        // Attempting to retrieve the size of a variable length field (e.g. Varchar) yields `None`.
        assert!(with_type(Type::Varchar).size().is_none());
    }

    #[test]
    fn test_get_type() {
        for ty in [Type::Null, Type::Boolean, Type::Integer, Type::Float] {
            assert_eq!(with_type(ty).field_type(), ty);
        }
    }

    #[test]
    fn test_equality() {
        let column = Column::new("TestColumn".to_string(), Type::Integer);

        // Two columns are equal if, and only if, all their fields are the same.
        let is_equal = column.clone();
        assert_eq!(is_equal, column);
        let name_is_different = Column::new("OtherColumn".to_string(), Type::Integer);
        assert_ne!(name_is_different, column);

        // In particular, note that two columns with matching names might not be equal!
        let type_is_different = Column::new("TestColumn".to_string(), Type::Float);
        assert_ne!(type_is_different, column);
    }

    fn with_type(field_type: Type) -> Column {
        Column {
            name: "TestColumn".to_string(),
            field_type,
        }
    }
}
