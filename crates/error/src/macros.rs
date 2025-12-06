/// Asserts the given expression errors on invocation.
#[macro_export]
macro_rules! assert_errors {
    ($f:expr) => {{
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $f));
        assert!(
            result.is_err(),
            "Expected an error, but the function succeeded."
        );
    }};
}

/// Constructs a [`crate::Error::InvalidData`] for the given format string.
#[macro_export]
macro_rules! errdata {
    ($($args:tt)*) => { $crate::Error::InvalidData(format!($($args)*)).into() };
}

/// Constructs a [`crate::Error::InvalidInput`] for the given format string.
#[macro_export]
macro_rules! errinput {
    ($($args:tt)*) => { $crate::Error::InvalidInput(format!($($args)*)).into() };
}
