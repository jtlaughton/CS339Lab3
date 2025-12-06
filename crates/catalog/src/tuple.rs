use bytes::Bytes;

#[derive(Debug)]
pub struct Tuple {
    data: Bytes,
}

impl Tuple {
    pub fn new(data: Bytes) -> Self {
        Self { data }
    }

    pub fn data(&self) -> Bytes {
        // Note this does not copy all the tuple data over into a new container; rather, it returns
        // a reference-counted pointer to the existing container, incrementing the reference count.
        Bytes::clone(&self.data)
    }

    pub fn tuple_size(&self) -> usize {
        self.data.len()
    }
}
