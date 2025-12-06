use crate::field::Field;
use crate::schema::Schema;
use crate::types::Type;

/// A utility struct that provides a mapping between serialized tuple data (e.g. &[u8]) and its
/// deserialized, semantically meaningful counterpart: `Vec<Field>`. Deserialization requires a
/// schema, which itself is an instruction set for how to interpret the bytes of a given payload.
///
/// A tuple, when represented as a list of fields, is serialized into `data: Vec<u8>` as follows:
///     ----------------------------------------------------------------------
///     | FIXED-SIZE or VARIED-SIZED OFFSET | PAYLOAD OF VARIED-SIZED FIELDS |
///     ----------------------------------------------------------------------
/// where all the fixed size fields are serialized and placed in the front, and any variable-length
/// field is placed _after_ the fixed sized field section, with the offset to that location in the
/// payload stored (as a serialized integer) with the fixed-size fields in order.
///
/// For example, a tuple [1, "hello", 3] with schema (INTEGER, VARCHAR, INTEGER) would be
/// serialized as follows:
///
///     1_i32   -> [1, 0, 0, 0] (in little endian)
///     "hello" -> [104, 101, 108, 108, 111]
///     3_i32   -> [3, 0, 0, 0] (in little endian)
///
/// ==> [1, 0, 0, 0, 12, 0, 0, 0, 3, 0, 0, 0, 104, 101, 108, 108, 111]
///          ^            ^           ^                  ^
///          |            |           |                  |
///        1_i32   the offset of     3_i32             "hello"
///                "hello" (12)
pub struct Serde {}
impl Serde {
    pub fn serialize(row: &[Field]) -> Vec<u8> {
        let fixed_payload_size = row
            .iter()
            .map(|field| field.get_type().size())
            .sum::<usize>();

        let bytes = {
            let mut bytes = Vec::with_capacity(fixed_payload_size);
            let mut var_len_offset = fixed_payload_size;
            let mut var_len_fields = Vec::new();

            // Build the fixed payload:
            for field in row {
                match field {
                    // For variable-length fields, add the offset to the payload now and the
                    // serialized field later.
                    Field::Varchar(_) => {
                        bytes.extend(var_len_offset.to_le_bytes());

                        let serialized_field = field.to_bytes();
                        var_len_offset += serialized_field.len();
                        var_len_fields.extend(serialized_field);
                    }
                    // For fixed-size fields, just add its serialized form to the payload.
                    _ => {
                        bytes.append(&mut field.to_bytes());
                    }
                }
            }
            // Now, add the serialized variable length fields to the end of the payload.
            bytes.extend(var_len_fields);
            bytes
        };

        bytes
    }

    pub fn deserialize(bytes: &[u8], schema: &Schema) -> Vec<Field> {
        let mut fields = Vec::with_capacity(schema.num_columns());
        // List of (index, offset) pairs, where an index `i` is the i-th field of the row, and
        // its corresponding offset is the serialized field's offset into the `bytes` payload.
        let mut varchar_offsets: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;

        for column in schema.columns() {
            match column.field_type() {
                Type::Null => {
                    fields.push(Field::Null);
                }
                Type::Varchar => {
                    let size = size_of::<usize>();
                    let offset = usize::from_le_bytes(bytes[i..i + size].try_into().unwrap());

                    varchar_offsets.push((fields.len(), offset));
                    // Push a dummy field into the fields vec for now to maintain the ordering.
                    fields.push(Field::Varchar("".to_string()));

                    i += size;
                }
                ty @ _ => {
                    let size = ty.size();
                    fields.push(Field::from_bytes(&bytes[i..i + size], ty));
                    i += size;
                }
            }
        }

        // Replace dummy varchar fields, if any exist, with their real values.
        for (n, (i, offset)) in varchar_offsets.iter().enumerate() {
            assert!(*i < fields.len());
            if n == varchar_offsets.len() - 1 {
                fields[*i] = Field::from_bytes(&bytes[*offset..], Type::Varchar);
            } else {
                let (_, next_offset) = varchar_offsets[n + 1];
                fields[*i] = Field::from_bytes(&bytes[*offset..next_offset], Type::Varchar);
            }
        }

        fields
    }
}

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::field::Field;
    use crate::schema::Schema;
    use crate::serde::Serde;
    use crate::types::Type;

    #[test]
    fn test_serde() {
        let schema = Schema::new(&columns_from(vec![
            Type::Integer,
            Type::Null,
            Type::Boolean,
            Type::Varchar,
            Type::Float,
        ]));
        let tuple = vec![
            Field::Integer(-34),
            Field::Null,
            Field::Boolean(false),
            Field::Varchar("hello".to_string()),
            Field::Float(f64::NEG_INFINITY),
        ];

        let serialized_tuple = Serde::serialize(&tuple);
        let deserialized_tuple = Serde::deserialize(&serialized_tuple, &schema);
        assert_eq!(tuple, deserialized_tuple);
    }

    fn columns_from(types: Vec<Type>) -> Vec<Column> {
        types
            .iter()
            .enumerate()
            .map(|(i, ty)| Column::new(i.to_string(), ty.clone()))
            .collect()
    }
}
