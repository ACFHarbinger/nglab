/*!
 * Generic traits for web scraping.
 */

use crate::errors::ArenaResult;

/**
 * Interface for components that scrape data from the web.
 */
pub trait WebScraper {
    /**
     * Download data for selected targets and save to a CSV file.
     *
     * @param output_path Path where the CSV will be saved.
     */
    async fn download_csv(&self, output_path: &str) -> ArenaResult<()>;
}
