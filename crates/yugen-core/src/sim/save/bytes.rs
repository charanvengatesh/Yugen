//! A bounds-checked forward reader.
//!
//! One type, shared by the two variable-length formats. The chunk format does
//! not use it and does not need to: its length is fixed and checked once, so a
//! truncated chunk falls out at that check rather than at the field it ran out
//! on. A run file and a world's identity file are both variable, and both are
//! read field by field off bytes another program may have written.
//!
//! Every accessor returns `Option`, so a truncated file becomes `None` at the
//! first short read instead of panicking somewhere deeper. That is the whole
//! design, and `no_truncation_of_a_run_decodes_to_something` is the test that
//! every prefix of a valid file stays refused.

/// A bounds-checked forward reader.
///
/// Every accessor returns `Option`, so a truncated file falls out as `None` at
/// the first short read instead of panicking somewhere deeper. The chunk format
/// does not need this — its length is fixed and checked once — and a run file's
/// is not.
pub(super) struct Reader<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) at: usize,
}

impl Reader<'_> {
    pub(super) fn take(&mut self, n: usize) -> Option<&[u8]> {
        let out = self.bytes.get(self.at..self.at + n)?;
        self.at += n;
        Some(out)
    }
    pub(super) fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    pub(super) fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }
    pub(super) fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    pub(super) fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    pub(super) fn f32(&mut self) -> Option<f32> {
        Some(f32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
}
