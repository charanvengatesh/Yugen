//! The editor, as a window.
//!
//! Everything it does lives in `yugen_editor::app`; this is the eleven lines that
//! find `content/` and open a window on it. Run it from anywhere in the tree:
//!
//! ```text
//! cargo run -p yugen-editor
//! cargo run -p yugen-editor -- path/to/content
//! ```

use std::path::PathBuf;

use yugen_editor::app::Editor;

fn main() -> eframe::Result<()> {
    // An explicit path wins; otherwise walk up from the working directory, which
    // is what makes `cargo run -p yugen-editor` work from any crate in the tree.
    let root = match std::env::args().nth(1) {
        Some(arg) => PathBuf::from(arg),
        None => {
            let here = std::env::current_dir().expect("a working directory");
            match Editor::find_content(&here) {
                Some(root) => root,
                None => {
                    eprintln!(
                        "no `content/` with a LAYOUT.toml above {} — pass one as an argument",
                        here.display()
                    );
                    std::process::exit(1);
                }
            }
        }
    };

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([1280.0, 820.0]),
        ..Default::default()
    };
    eframe::run_native(
        "yugen — content editor",
        options,
        Box::new(move |_cc| Ok(Box::new(Editor::new(root)))),
    )
}
