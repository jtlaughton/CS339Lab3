use crate::types::Type;

/// Represents a view over a SQL value data stored in some materialized state. Normally, tuple data
/// is passed around as a byte slice (e.g. data: Vec<u8>); you can think of this `Field` class as
/// being the value type the tuple data consists of once it's deserialized with its corresponding
/// schema.
///
/// For example, say we have a tuple `(false, 1, 2)` with schema `(Boolean, Integer, Integer)`.
/// We could serialize the tuple as
/// ```
///  use bytes::BufMut;
///  let data = {
///     // bool (1 byte) + i32 (4 bytes) + i32 (4 bytes) = 9 bytes.
///     let mut payload: Vec<u8> = Vec::with_capacity(9);
///     payload.put_bytes(u8::from(false), 1);  // Boolean: false
///     payload.put_i32(1);                     // Integer: 1
///     payload.put_i32(2);                     // Integer: 2
///
///     payload
///  };
/// ```
/// and then pass it around as a byte slice `&[u8]`, which is cool. But, at some point, we'll want
/// to run queries over the actual field values of the tuple, e.g. false, 1, 2. When we do that,
/// we'll need to materialize the field values from the tuple data into some sort of value object;
/// this `Field` object provides a way to do so.
#[derive(Debug, Clone)]
pub enum Field {
    Null,
    Boolean(bool),
    Integer(i32),
    Float(f64),
    Varchar(String),
}

impl Field {
    /// Serializes a field into an owned byte slice.
    ///
    /// Note that [`Field::Float`] and [`Field::Integer`] values get serialized into their byte
    /// representation in **little-endian** form!
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Field::Null => vec![],
            Field::Boolean(value) => vec![u8::from(*value)],
            Field::Integer(value) => Vec::from(i32::to_le_bytes(*value)),
            Field::Float(value) => Vec::from(f64::to_le_bytes(*value)),
            Field::Varchar(string) => string.as_bytes().to_vec(),
        }
    }

    /// Deserializes a byte slice into a field, given the field's underlying data type, which is
    /// represented by a [`crate::catalog::types::Type`].
    ///
    /// Remember that [`Field::Float`] and [`Field::Integer`] are represented as **little-endian**
    /// byte slices!
    pub fn from_bytes(bytes: &[u8], field_type: Type) -> Self {
        if field_type != Type::Varchar {
            assert_eq!(field_type.size(), bytes.len());
        }
        match field_type {
            Type::Null => Field::Null,
            Type::Boolean => Field::Boolean(bytes[0] == 1),
            Type::Integer => Field::Integer(i32::from_le_bytes(bytes.try_into().unwrap())),
            Type::Float => Field::Float(f64::from_le_bytes(bytes.try_into().unwrap())),
            Type::Varchar => Field::Varchar(String::from_utf8(bytes.to_vec()).unwrap()),
        }
    }

    /// Returns the corresponding [`crate::types::Type`] for the given field.
    pub fn get_type(&self) -> Type {
        match self {
            Field::Null => Type::Null,
            Field::Boolean(_) => Type::Boolean,
            Field::Integer(_) => Type::Integer,
            Field::Float(_) => Type::Float,
            Field::Varchar(_) => Type::Varchar,
        }
    }
}

impl Default for Field {
    fn default() -> Self {
        Field::Null
    }
}

impl From<bool> for Field {
    fn from(b: bool) -> Self {
        Field::Boolean(b)
    }
}

impl From<i32> for Field {
    fn from(i: i32) -> Self {
        Field::Integer(i)
    }
}

impl From<f64> for Field {
    fn from(f: f64) -> Self {
        Field::Float(f)
    }
}

impl From<&str> for Field {
    fn from(s: &str) -> Self {
        Field::Varchar(s.to_owned())
    }
}

impl PartialEq for Field {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Field::Null, Field::Null) => true,
            (Field::Boolean(a), Field::Boolean(b)) => a.eq(b),
            (Field::Integer(a), Field::Integer(b)) => a.eq(b),
            (Field::Varchar(a), Field::Varchar(b)) => a.eq(b),
            (Field::Float(a), Field::Float(b)) => {
                // Match on NaN, in addition to equality, for floats.
                a.eq(b) || (a.is_nan() && b.is_nan())
            }
            _ => false,
        }
    }
}
impl Eq for Field {}

impl Ord for Field {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Field::Null, Field::Null) => std::cmp::Ordering::Equal,
            // Nothing is less than something.
            (Field::Null, _) => std::cmp::Ordering::Less,
            // Something is greater than nothing.
            (_, Field::Null) => std::cmp::Ordering::Greater,
            // Comparable values just get compared as expected.
            (Field::Boolean(a), Field::Boolean(b)) => a.cmp(b),
            (Field::Integer(a), Field::Integer(b)) => a.cmp(b),
            (Field::Varchar(a), Field::Varchar(b)) => a.cmp(b),
            // Per IEEE standard, NaN should not be comparable to anything (including itself).
            // But we still might need to query for it, so we're going to pretend it's comparable.
            (Field::Float(a), Field::Float(b)) => match (a.is_nan(), b.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => a.partial_cmp(b).unwrap(),
            },
            _ => unimplemented!(
                "Different value types should not be compared, with the exception of NULL."
            ),
        }
    }
}
impl PartialOrd for Field {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::ops::Add for Field {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        match (self, other) {
            (Field::Integer(l), Field::Integer(r)) => {
                l.checked_add(r).map_or(Field::Null, Field::Integer)
            }
            (Field::Integer(l), Field::Float(r)) => Field::Float((l as f64).add(r)),
            (Field::Float(l), Field::Integer(r)) => Field::Float(l.add(r as f64)),
            (Field::Float(l), Field::Float(r)) => Field::Float(l.add(r)),
            // We shouldn't be able to add non-numerical types.
            _ => Field::Null,
        }
    }
}

impl std::ops::Sub for Field {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        match (self, other) {
            (Field::Integer(l), Field::Integer(r)) => {
                l.checked_sub(r).map_or(Field::Null, Field::Integer)
            }
            (Field::Integer(l), Field::Float(r)) => Field::Float((l as f64).sub(r)),
            (Field::Float(l), Field::Integer(r)) => Field::Float(l.sub(r as f64)),
            (Field::Float(l), Field::Float(r)) => Field::Float(l.sub(r)),
            // We shouldn't be able to subtract non-numerical types.
            _ => Field::Null,
        }
    }
}

impl std::ops::Mul for Field {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        match (self, other) {
            (Field::Integer(l), Field::Integer(r)) => {
                l.checked_mul(r).map_or(Field::Null, Field::Integer)
            }
            (Field::Integer(l), Field::Float(r)) => Field::Float((l as f64).mul(r)),
            (Field::Float(l), Field::Integer(r)) => Field::Float(l.mul(r as f64)),
            (Field::Float(l), Field::Float(r)) => Field::Float(l.mul(r)),
            // We shouldn't be able to multiply non-numerical types.
            _ => Field::Null,
        }
    }
}

impl std::ops::Div for Field {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        match (self, other) {
            (Field::Integer(l), Field::Integer(r)) => {
                l.checked_div(r).map_or(Field::Null, Field::Integer)
            }
            (Field::Integer(l), Field::Float(r)) => Field::Float((l as f64).div(r)),
            (Field::Float(l), Field::Integer(r)) => Field::Float(l.div(r as f64)),
            (Field::Float(l), Field::Float(r)) => Field::Float(l.div(r)),
            // We shouldn't be able to divide non-numerical types.
            _ => Field::Null,
        }
    }
}

impl std::ops::Rem for Field {
    type Output = Self;
    fn rem(self, other: Self) -> Self {
        match (self, other) {
            (Field::Integer(l), Field::Integer(r)) => {
                l.checked_rem(r).map_or(Field::Null, Field::Integer)
            }
            (Field::Integer(l), Field::Float(r)) => Field::Float((l as f64).rem(r)),
            (Field::Float(l), Field::Integer(r)) => Field::Float(l.rem(r as f64)),
            (Field::Float(l), Field::Float(r)) => Field::Float(l.rem(r)),
            // We shouldn't be able to mod non-numerical types.
            _ => Field::Null,
        }
    }
}

impl std::fmt::Display for Field {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Null => write!(f, "NULL"),
            Self::Boolean(true) => write!(f, "TRUE"),
            Self::Boolean(false) => write!(f, "FALSE"),
            Self::Integer(i) => i.fmt(f),
            Self::Float(float) => float.fmt(f),
            Self::Varchar(varchar) => write!(f, "{}", varchar.escape_default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::field::Field;
    use crate::types::Type;
    use rustdb_error::assert_errors;

    #[test]
    fn test_creation() {
        // Default initialization should be NULL.
        let field = Field::default();
        assert_eq!(Field::Null, field);

        // Booleans properly initialized.
        assert_eq!(Field::Boolean(true), true.into());
        assert_eq!(Field::Boolean(false), false.into());

        // Typical and MAX/MIN integers properly initialized.
        assert_eq!(Field::Integer(0), 0.into());
        assert_eq!(Field::Integer(-32), (-32).into());
        assert_eq!(Field::Integer(i32::MIN), i32::MIN.into());
        assert_eq!(Field::Integer(i32::MAX), i32::MAX.into());

        // Typical, MAX/MIN, INFINITY, and NaN floats properly initialized.
        for f in [
            0.0,
            -0.0,
            339.2,
            f64::MAX,
            f64::MIN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            assert_eq!(Field::Float(f), Field::Float(f));
        }

        // Text strings properly initialized. (Can't forget about the emojis...)
        for text in ["Hello, world", "", "All love 🛸💕🕺", "339", "1 of 1"] {
            assert_eq!(Field::Varchar(text.into()), text.into());
        }
    }

    #[test]
    fn test_equality() {
        // Nulls should be equal to each other...
        assert_eq!(Field::Null, Field::Null);
        // ...and non-equal to everything else.
        assert_ne!(Field::Null, Field::Boolean(true));
        assert_ne!(Field::Null, Field::Boolean(false));
        assert_ne!(Field::Null, Field::Integer(0));
        assert_ne!(Field::Null, Field::Float(0.0));
        assert_ne!(Field::Null, Field::Varchar("".into()));

        // Booleans should only be equal to other booleans of the same parity...
        assert_eq!(Field::Boolean(true), Field::Boolean(true));
        assert_eq!(Field::Boolean(false), Field::Boolean(false));
        assert_ne!(Field::Boolean(true), Field::Boolean(false));
        assert_ne!(Field::Boolean(false), Field::Boolean(true));
        // ...and definitely not equal to fields of other types.
        assert_ne!(Field::Boolean(false), Field::Integer(0));
        assert_ne!(Field::Boolean(false), Field::Float(0.0));
        assert_ne!(Field::Boolean(false), Field::Varchar("".into()));

        // Integers should be equal to only other integer fields with the same value.
        let ints = [0, 1, -32, i32::MAX, i32::MIN];
        for int in ints {
            assert_eq!(Field::Integer(int), Field::Integer(int));
        }
        for window in ints.windows(2) {
            assert_ne!(Field::Integer(window[0]), Field::Integer(window[1]));
        }
        // Integer field types aren't implicitly comparable to Float or Varchar types.
        assert_ne!(Field::Integer(0), Field::Float(0.0));
        assert_ne!(Field::Integer(0), Field::Varchar("0".into()));

        // Floats match on NaN in addition to equality, as we want to be able to query for NaN's.
        let floats = [
            0.0,
            -32.4,
            f64::MIN,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for float in floats {
            assert_eq!(Field::Float(float), Field::Float(float));
        }
        for window in floats.windows(2) {
            assert_ne!(Field::Float(window[0]), Field::Float(window[1]));
        }
        // Float field types aren't implicitly comparable to Varchar types.
        assert_ne!(
            Field::Float(12.34567890),
            Field::Varchar("12.34567890".into())
        );

        // Varchar equality based on string members.
        let strings = ["Hello, world", "", "All love 🛸💕🕺", "339", "1 of 1"];
        for text in strings {
            assert_eq!(Field::Varchar(text.into()), text.into());
        }
        for window in strings.windows(2) {
            assert_ne!(
                Field::Varchar(window[0].to_owned()),
                Field::Varchar(window[1].to_owned())
            );
        }
    }

    #[test]
    fn test_comparison() {
        // NULL is less than any non-null field.
        assert!(Field::Boolean(false) > Field::Null);
        assert!(Field::Integer(0) > Field::Null);
        assert!(Field::Null < Field::Float(f64::NEG_INFINITY));
        assert!(Field::Null < Field::Varchar("".into()));

        // Integer and float comparison works as expected.
        assert!(Field::Integer(-1) < Field::Integer(2));
        assert!(Field::Integer(2) > Field::Integer(1));
        assert!(Field::Float(0.0) < Field::Float(1.2));
        assert!(Field::Float(0.0) > Field::Float(-1.2));

        // Attempting to compare non-null fields of different data types will throw an exception.
        assert_errors!(Field::Boolean(false) < Field::Integer(0));
        assert_errors!(Field::Integer(0) < Field::Float(0.0));
        assert_errors!(Field::Float(0.0) < Field::Varchar("0".into()));
    }

    /// Given Serialization (`Ser: Field -> [u8]`) and deserialization (`De: [u8] -> Field`), we
    /// can assume correctness if it can be shown that deserialization is an inverse mapping of
    /// serialization, i.e. `De(Ser(x)) = x`.
    ///
    /// This is because in the context of our DBMS, we'll always start with a field type, later
    /// deserializing it when it needs to be passed around as a payload of bytes. In our case,
    /// we'll never construct a sequence of bytes, and THEN materialize it to a field if the
    /// field never existed prior to the bytes.
    #[test]
    fn test_serialization() {
        let null = Field::Null;
        assert_eq!(Field::from_bytes(&null.to_bytes(), Type::Null), null);

        [Field::Boolean(true), Field::Boolean(false)]
            .iter()
            .for_each(|bool_field| {
                assert_eq!(
                    Field::from_bytes(&bool_field.to_bytes(), Type::Boolean),
                    *bool_field
                )
            });

        [0, 123456789, -123456789, i32::MAX, i32::MIN]
            .map(|i| Field::Integer(i))
            .iter()
            .for_each(|int_field| {
                assert_eq!(
                    Field::from_bytes(&int_field.to_bytes(), Type::Integer),
                    *int_field
                )
            });

        [
            0.0,
            -1.0,
            339.339,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NAN,
            f64::MAX,
            f64::MIN,
        ]
        .map(|i| Field::Float(i))
        .iter()
        .for_each(|float_field| {
            assert_eq!(
                Field::from_bytes(&float_field.to_bytes(), Type::Float),
                *float_field
            )
        });

        ["Hello, world", "", "All love 🛸💕🕺", "339", "1 of 1"]
            .map(|text| Field::Varchar(text.into()))
            .iter()
            .for_each(|varchar_field| {
                assert_eq!(
                    Field::from_bytes(&varchar_field.to_bytes(), Type::Varchar),
                    *varchar_field,
                )
            });
    }
}
