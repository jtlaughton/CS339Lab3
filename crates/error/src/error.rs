pub type Result<T> = std::result::Result<T, Error>;
impl<T> From<Error> for Result<T> {
    fn from(e: Error) -> Self {
        Err(e)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// Invalid data, which typically includes decoding errors or unexpected internal values.
    InvalidData(String),
    /// Invalid user input, which typically includes parser or query errors.
    InvalidInput(String),
    /// An IO error has occurred.
    IO(String),
    /// A numerical error has occurred, such as an integer overflow.
    ArithmeticOverflow,
    /// An out-of-bounds access has occurred.
    OutOfBounds,
    /// A buffer pool error has occured.
    BufferPoolError(String),
    /// The page cannot be deleted because it is still pinned.
    PagePinned(u32),
}

impl std::error::Error for Error {}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            Error::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            Error::IO(msg) => write!(f, "IO error: {}", msg),
            Error::ArithmeticOverflow => write!(f, "Arithmetic overflow"),
            Error::OutOfBounds => write!(f, "Out of bounds"),
            Error::BufferPoolError(msg) => write!(f, "Buffer error: {}", msg),
            Error::PagePinned(page_id) => {
                write!(f, "Cannot delete page {}: Page is still pinned", page_id)
            }
        }
    }
}

impl serde::de::Error for Error {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Error::InvalidData(msg.to_string())
    }
}

impl serde::ser::Error for Error {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Error::InvalidInput(msg.to_string())
    }
}

impl From<std::array::TryFromSliceError> for Error {
    fn from(e: std::array::TryFromSliceError) -> Self {
        Error::InvalidData(e.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::IO(e.to_string())
    }
}

impl From<std::num::ParseIntError> for Error {
    fn from(e: std::num::ParseIntError) -> Self {
        Error::InvalidInput(e.to_string())
    }
}

impl From<std::num::ParseFloatError> for Error {
    fn from(e: std::num::ParseFloatError) -> Self {
        Error::InvalidInput(e.to_string())
    }
}

impl From<std::num::TryFromIntError> for Error {
    fn from(e: std::num::TryFromIntError) -> Self {
        Error::InvalidData(e.to_string())
    }
}

impl From<std::string::FromUtf8Error> for Error {
    fn from(e: std::string::FromUtf8Error) -> Self {
        Error::InvalidData(e.to_string())
    }
}

impl<T> From<std::sync::PoisonError<T>> for Error {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        // This occurs when a different thread panics when holding a mutex. Since this is
        // fatal, we should panic here too.
        panic!("{e}")
    }
}
