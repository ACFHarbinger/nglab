/*!
 * Generic traits for web scraping.
 */

use crate::errors::ArenaResult;

/**
 * Interface for components that scrape data from the web.
 *
 * Note: `async fn` in traits is generally discouraged in public interfaces
 * because it inhibits auto traits like `Send`. However, for internal usage
 * or where `Send` is not strictly required across thread bounds for the future itself,
 * we suppress this lint.
 */
#[allow(async_fn_in_trait)]
pub trait WebScraper {
    /**
     * Download data for selected targets and save to a CSV file.
     *
     * @param output_path Path where the CSV will be saved.
     */
    async fn download_csv(&self, output_path: &str) -> ArenaResult<()>;
}
