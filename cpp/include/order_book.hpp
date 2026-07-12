// Data-Oriented Design (DOD) order book for the HFT Native Loop (Tier 1).
//
// Layout is struct-of-arrays: parallel arrays indexed by a dense price-level
// handle, kept contiguous to eliminate CPU cache misses on the hot path. Orders
// reference levels by index (not pointer). No dynamic allocation in steady state
// — capacities are reserved up front. See moon/roadmaps/hft_cpp.md §1 and
// .agent/skills/add-hft-strategy.md.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nglab::hft {

enum class Side : uint8_t { Bid = 0, Ask = 1 };

// A minimal DOD price-level book. Prices are integer ticks (fixed-point) to keep
// comparisons branch-friendly and exact.
class OrderBook {
public:
    explicit OrderBook(std::size_t capacity = 1u << 16) {
        price_.reserve(capacity);
        qty_.reserve(capacity);
        side_.reserve(capacity);
    }

    // Add resting liquidity at an integer-tick price. Returns the level handle.
    std::uint32_t add(Side side, std::int64_t price_ticks, std::int64_t qty) {
        const auto handle = static_cast<std::uint32_t>(price_.size());
        price_.push_back(price_ticks);
        qty_.push_back(qty);
        side_.push_back(side);
        return handle;
    }

    // Best bid / ask in integer ticks. Returns false if that side is empty.
    bool best_bid(std::int64_t& out) const noexcept { return best(Side::Bid, true, out); }
    bool best_ask(std::int64_t& out) const noexcept { return best(Side::Ask, false, out); }

    // Match an incoming aggressive order against the opposite side, consuming
    // resting quantity greedily. Returns the filled quantity.
    std::int64_t match(Side taker, std::int64_t price_ticks, std::int64_t qty) noexcept;

    std::size_t levels() const noexcept { return price_.size(); }

private:
    bool best(Side side, bool want_max, std::int64_t& out) const noexcept;

    // Struct-of-arrays hot storage.
    std::vector<std::int64_t> price_;
    std::vector<std::int64_t> qty_;
    std::vector<Side>         side_;
};

}  // namespace nglab::hft
