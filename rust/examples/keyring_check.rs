fn main() {
    let service = "nglab";
    let username = "debug_user_123";
    let entry = keyring::Entry::new(service, username).unwrap();

    // We can't easily get the backend name from Entry in v3 without specialization,
    // but we can try to use different methods to see which one works.
    println!("Testing keyring backend for {} / {}", service, username);

    // Check for session/system bus
    // (This is internal to the crate, but we can try to see if it works at all)
    match entry.get_password() {
        Ok(p) => println!("Got password: {}", p),
        Err(e) => println!("Get password error: {}", e),
    }
}
