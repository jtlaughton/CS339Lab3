/// An exhaustive enumeration of all the data types of a [`crate::catalog::field::Field`] object.
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub enum Type {
    Null,
    Boolean,
    Integer,
    Float,
    Varchar,
}

impl Type {
    /// Returns the byte size of this type if it's fixed size; otherwise, for variable-length types
    /// returns the byte size of its offset into the tuple data payload (i.e. size_of(usize)).
    pub fn size(&self) -> usize {
        match self {
            Type::Null => 0,
            Type::Boolean => 1,
            // We work with i32's, which are 4 bytes.
            Type::Integer => 4,
            // We work with f64's, which are 8 bytes.
            Type::Float => 8,
            // Strings are variable-length, so inferring the size from this enum is impossible.
            Type::Varchar => size_of::<usize>(),
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", format!("{:?}", self))
    }
}
