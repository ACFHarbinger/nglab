use keyring::Entry;

fn main() {
    let service = "nglab";
    let username = "debug_user_123";
    let entry = Entry::new(service, username).unwrap();

    println!("Creating entry for {} / {}", service, username);
    entry.set_password("debug_password").unwrap();

    println!("Entry created. Please run: secret-tool search --all service nglab");
}
