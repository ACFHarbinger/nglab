use keyring::Entry;

fn main() {
    let service = "nglab";
    // Since we don't have a list of entries, we'll try to delete common ones
    // and also try to clear the debug entry.
    let candidates = vec!["acfharbinger", "debug_user_123", "test_user"];

    for username in candidates {
        let entry = Entry::new(service, username).unwrap();
        match entry.get_password() {
            Ok(_) => {
                println!("Found entry for {}. Deleting...", username);
                match entry.delete_credential() {
                    Ok(_) => println!("✅ Deleted {}", username),
                    Err(e) => println!("❌ Failed to delete {}: {}", username, e),
                }
            }
            Err(_) => {
                // Not found, check for the debug one
            }
        }
    }
    println!("Nuke complete. If login still works, the data is likely not in the keyring.");
}
