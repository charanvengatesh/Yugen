//! The one error type. Everything in the compiler throws it, the CLI catches it.

use std::fmt;
use std::rc::Rc;

/// Source position, carried on every token so errors can point at a file:line.
///
/// `file` is an `Rc<str>` because every token in a file shares it and the
/// compiler produces tens of thousands of them; the TS version got this for free
/// from string interning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loc {
    pub file: Rc<str>,
    pub line: usize,
}

impl Loc {
    pub fn new(file: &Rc<str>, line: usize) -> Self {
        Loc {
            file: Rc::clone(file),
            line,
        }
    }
}

impl fmt::Display for Loc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.line)
    }
}

/// A compile error that knows where it came from.
#[derive(Clone, Debug)]
pub struct ContentError {
    pub message: String,
    pub loc: Option<Loc>,
}

impl ContentError {
    pub fn new(message: impl Into<String>) -> Self {
        ContentError {
            message: message.into(),
            loc: None,
        }
    }

    pub fn at(message: impl Into<String>, loc: &Loc) -> Self {
        ContentError {
            message: message.into(),
            loc: Some(loc.clone()),
        }
    }

    /// Attach a location to an error raised somewhere that did not have one.
    /// Never overwrites a location that is already there — the innermost site
    /// knows best.
    pub fn or_at(mut self, loc: &Loc) -> Self {
        if self.loc.is_none() {
            self.loc = Some(loc.clone());
        }
        self
    }
}

impl fmt::Display for ContentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.loc {
            Some(loc) => write!(f, "{loc}: {}", self.message),
            None => write!(f, "{}", self.message),
        }
    }
}

impl std::error::Error for ContentError {}

pub type Result<T> = std::result::Result<T, ContentError>;

/// `err!("bad thing: {x}")` / `err!(&loc, "bad thing: {x}")`
///
/// The located and unlocated forms are told apart by whether the first token is
/// a string literal, which is why the literal rule has to come first.
#[macro_export]
macro_rules! err {
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::error::ContentError::new(format!($fmt $(, $arg)*))
    };
    ($loc:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::error::ContentError::at(format!($fmt $(, $arg)*), $loc)
    };
}

/// `bail!(...)` — the same, but returns.
#[macro_export]
macro_rules! bail {
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        return Err($crate::error::ContentError::new(format!($fmt $(, $arg)*)))
    };
    ($loc:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {
        return Err($crate::error::ContentError::at(format!($fmt $(, $arg)*), $loc))
    };
}
