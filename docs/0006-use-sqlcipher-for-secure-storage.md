# ADR-0006: Use SQLCipher for Secure Storage

## Status
Accepted

## Context
NGLab handles sensitive user credentials (e.g., Polymarket API keys, Wallet private keys) and potentially proprietary trading strategy parameters. Storing these in plain text on the user's filesystem is a significant security risk. We need a robust, local storage solution that offers strong encryption at rest.

## Decision
We will use **SQLCipher** (via the `rusqlite` crate with the `bundled-sqlcipher` feature) as our secure storage backend.
- **Encryption**: AES-256 encryption for the entire database file.
- **Key Derivation**: We will use Argon2id to derive the encryption key from a user-supplied master password.
- **Integration**: The `VaultManager` struct in `rust/src/secret/vault.rs` will encapsulate all database interactions.

## Consequences
- **Easier**:
    - **Compliance**: Meets standard security requirements for credential storage.
    - **Portability**: The database is a single encrypted file that can be easily backed up or moved (if the master password is known).
    - **Querying**: We can use standard SQL for data management once the secure connection is established.
- **Difficult**:
    - **Dependency**: Requires linking against the SQLCipher C library. Using the `bundled` feature simplifies this but increases build times.
    - **Performance**: Encryption/decryption adds overhead to every I/O operation, though negligible for configuration usage.
    - **Recovery**: If the user loses their master password, the data is unrecoverable.

## Alternatives Considered
- **OS Keyring (libsecret/KeyChain)**: Good for small items like passwords, but clunky for structured data (like strategy configurations or trade history). We use the OS keyring only to store the *salt* for the master password, not the data itself.
- **Plain JSON + GPG**: Valid, but requires external tools and parsing the entire file to read one record. SQLCipher offers random access.
- **Symmetric File Encryption**: Manually encrypting files with `AES-GCM`. Prone to implementation errors (nonce reuse, etc.). SQLCipher is a battle-tested industry standard.
