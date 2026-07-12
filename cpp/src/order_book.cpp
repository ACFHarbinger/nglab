#include "nglab_hft/order_book.hpp"

namespace nglab::hft {

bool OrderBook::best(Side side, bool want_max, std::int64_t& out) const noexcept {
    bool found = false;
    std::int64_t acc = 0;
    for (std::size_t i = 0; i < price_.size(); ++i) {
        if (side_[i] != side || qty_[i] <= 0) continue;
        if (!found || (want_max ? price_[i] > acc : price_[i] < acc)) {
            acc = price_[i];
            found = true;
        }
    }
    if (found) out = acc;
    return found;
}

std::int64_t OrderBook::match(Side taker, std::int64_t price_ticks, std::int64_t qty) noexcept {
    // Aggressive Bid crosses Asks priced <= limit; aggressive Ask crosses Bids
    // priced >= limit. Greedy consume; price-time priority is a follow-up.
    const Side maker = (taker == Side::Bid) ? Side::Ask : Side::Bid;
    std::int64_t remaining = qty;
    for (std::size_t i = 0; i < price_.size() && remaining > 0; ++i) {
        if (side_[i] != maker || qty_[i] <= 0) continue;
        const bool crosses = (taker == Side::Bid) ? (price_[i] <= price_ticks)
                                                   : (price_[i] >= price_ticks);
        if (!crosses) continue;
        const std::int64_t take = (qty_[i] < remaining) ? qty_[i] : remaining;
        qty_[i] -= take;
        remaining -= take;
    }
    return qty - remaining;  // filled
}

}  // namespace nglab::hft
