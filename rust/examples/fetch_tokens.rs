use nglab::web::polymarket::PolymarketScraper;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: fetch_tokens <market_slug>");
        return;
    }
    let slug = &args[1];
    let mut scraper = PolymarketScraper::new();
    match scraper.resolve_market(slug).await {
        Ok(_) => {
            let metadata = scraper.get_metadata();
            println!("Market: {}", metadata.title);
            for outcome in metadata.outcomes {
                println!("ID: {}, Name: {}", outcome.id, outcome.name);
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}
