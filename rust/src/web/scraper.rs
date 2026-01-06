use crate::error::ArenaResult;

/// Trait for web scrapers
pub trait WebScraper {
    /// Download data and save as CSV
    fn download_csv(&self, output_path: &str) -> ArenaResult<()>;
}
