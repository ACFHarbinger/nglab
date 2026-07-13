# Skill: Add / Change a Protobuf Message

All cross-boundary structures live in `proto/` and are code-generated for TS, Rust, Go, and C++.

1. **Edit the schema**: add/modify messages in `proto/*.proto` under the versioned package
   (`nglab.v1`). Additive-only within a major version (new fields with new tags; never reuse tags).
2. **Regenerate all four**: `just schema::gen` (prost → Rust, protoc-gen-go → Go, protoc C++ →
   `cpp/generated`, ts-proto → TS). CI fails if generated code drifts.
3. **Hot-path mirror**: if the message is read from C++ shared memory, update the POD mirror struct
   and keep its layout documented and in lockstep.
4. **Consumers**: update the Go loopback framing and the Rust shm/loopback readers to match.
5. **Docs**: update `moon/roadmaps/schema_protobuf.md`; note the schema version bump in CHANGELOG.
