/// <reference types="cypress" />

// Import commands
import "./commands";

// Mock Tauri API before each test
beforeEach(() => {
  // Intercept Tauri invoke calls by stubbing window.__TAURI__
  cy.window().then((win) => {
    // Create mock Tauri core module
    const mockInvoke = cy.stub().as("tauriInvoke");

    // Set up default responses for common commands
    mockInvoke.withArgs("get_public_polymarket_markets").resolves({
      success: true,
      message: "Found 10 markets",
      data: getMockMarkets(),
    });

    mockInvoke.withArgs("search_public_polymarket_markets").resolves({
      success: true,
      message: "Found 5 markets",
      data: getMockMarkets().slice(0, 5),
    });

    mockInvoke.withArgs("get_current_user").resolves(null);

    // Default fallback for unhandled commands
    mockInvoke.resolves({ success: true, data: null });

    // Inject mock into window
    (win as any).__TAURI__ = {
      core: {
        invoke: mockInvoke,
      },
    };

    // Also mock the @tauri-apps/api/core module pattern
    (win as any).__TAURI_INTERNALS__ = {
      invoke: mockInvoke,
    };
  });
});

// Helper function to generate mock market data
function getMockMarkets() {
  return [
    {
      id: "market-1",
      event_id: "event-1",
      title: "Will Bitcoin reach $100k?",
      question: "Will Bitcoin reach $100,000 by end of 2026?",
      outcomes: ["Yes", "No"],
      clob_token_ids: ["token-1-yes", "token-1-no"],
      volume: 5000000,
      liquidity: 250000,
      end_date: "2026-12-31",
      active: true,
    },
    {
      id: "market-2",
      event_id: "event-2",
      title: "Super Bowl Champion 2026",
      question: "Who will win the Super Bowl?",
      outcomes: ["Yes", "No"],
      clob_token_ids: ["token-2-yes", "token-2-no"],
      volume: 8000000,
      liquidity: 400000,
      end_date: "2026-02-08",
      active: true,
    },
    {
      id: "market-3",
      event_id: "event-3",
      title: "Fed Rate Decision",
      question: "Will the Fed cut rates in January?",
      outcomes: ["Yes", "No"],
      clob_token_ids: ["token-3-yes", "token-3-no"],
      volume: 3000000,
      liquidity: 150000,
      end_date: "2026-01-31",
      active: true,
    },
    {
      id: "market-4",
      event_id: "event-4",
      title: "Trump Approval Rating",
      question: "Will Trump's approval exceed 50%?",
      outcomes: ["Yes", "No"],
      clob_token_ids: ["token-4-yes", "token-4-no"],
      volume: 2500000,
      liquidity: 120000,
      end_date: "2026-06-30",
      active: true,
    },
    {
      id: "market-5",
      event_id: "event-5",
      title: "Ethereum Price Prediction",
      question: "Will ETH reach $10,000?",
      outcomes: ["Yes", "No"],
      clob_token_ids: ["token-5-yes", "token-5-no"],
      volume: 4500000,
      liquidity: 200000,
      end_date: "2026-12-31",
      active: true,
    },
  ];
}

// Export for use in tests
export { getMockMarkets };
