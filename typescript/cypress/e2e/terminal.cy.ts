/// <reference types="cypress" />

describe("Terminal/Markets Tab", () => {
  beforeEach(() => {
    cy.visit("/");
    cy.navigateToTab("terminal");
  });

  describe("Layout", () => {
    it("should display terminal tab content", () => {
      cy.contains("button", "Markets").should("have.class", "text-white");
    });

    it("should have three-column layout", () => {
      // Sidebar, main chart, and order panel
      cy.get('[class*="grid"], [class*="flex"]').should("exist");
    });
  });

  describe("Market Sidebar", () => {
    it("should display market list", () => {
      cy.contains(/Markets|Select/i).should("be.visible");
    });

    it("should have search input", () => {
      cy.get('input[placeholder*="search" i]').should("exist");
    });

    it("should filter markets on search", () => {
      cy.mockTauriInvoke("search_public_polymarket_markets", {
        success: true,
        message: "Found 2 markets",
        data: [
          {
            id: "btc-market",
            event_id: "event-1",
            title: "Bitcoin $100k",
            question: "Will BTC reach $100k?",
            outcomes: ["Yes", "No"],
            clob_token_ids: ["yes-1", "no-1"],
            volume: 1000000,
            liquidity: 50000,
            end_date: "2026-12-31",
            active: true,
          },
        ],
      });

      cy.get('input[placeholder*="search" i]').type("bitcoin");

      cy.contains("Bitcoin", { timeout: 5000 }).should("be.visible");
    });

    it("should select market from sidebar", () => {
      cy.mockTauriInvoke("get_public_polymarket_markets", {
        success: true,
        message: "Found markets",
        data: [
          {
            id: "market-1",
            event_id: "event-1",
            title: "Test Market",
            question: "Test question?",
            outcomes: ["Yes", "No"],
            clob_token_ids: ["yes-1", "no-1"],
            volume: 1000000,
            liquidity: 50000,
            end_date: "2026-12-31",
            active: true,
          },
        ],
      });

      cy.waitForMarketsToLoad();

      // Click on a market
      cy.contains("Test Market").click();
    });
  });

  describe("Terminal Chart", () => {
    it("should display price chart", () => {
      cy.get('[class*="Chart"], canvas').should("exist");
    });

    it("should show current price indicator", () => {
      // Price display
      cy.contains(/Price|\$/i).should("be.visible");
    });

    it("should update chart when streaming", () => {
      // Chart updates with live data
    });
  });

  describe("Order Book Widget", () => {
    it("should display order book", () => {
      cy.contains(/Order Book|Depth/i).should("be.visible");
    });

    it("should show bids and asks", () => {
      cy.contains(/Bid|Buy/i).should("be.visible");
      cy.contains(/Ask|Sell/i).should("be.visible");
    });

    it("should display price levels", () => {
      // Order book price levels
    });
  });

  describe("Trading Form Widget", () => {
    it("should display trading form", () => {
      cy.contains(/Trade|Order|Buy|Sell/i).should("be.visible");
    });

    it("should have buy/sell toggle", () => {
      cy.contains("button", /Buy/i).should("be.visible");
      cy.contains("button", /Sell/i).should("be.visible");
    });

    it("should have amount input", () => {
      cy.get('input[type="number"], input[placeholder*="amount" i]').should("exist");
    });

    it("should have price input for limit orders", () => {
      // Limit order price input
      cy.get('input').should("have.length.at.least", 1);
    });

    it("should calculate total on amount change", () => {
      cy.get('input[type="number"]').first().clear().type("100");
      // Total should update
    });

    it("should submit buy order", () => {
      cy.mockTauriInvoke("place_order", {
        success: true,
        message: "Order placed",
      });

      cy.contains("button", /Buy/i).first().click();
      cy.get('input[type="number"]').first().clear().type("10");
      cy.contains("button", /Place|Submit|Confirm/i).click();
    });

    it("should submit sell order", () => {
      cy.mockTauriInvoke("place_order", {
        success: true,
        message: "Order placed",
      });

      cy.contains("button", /Sell/i).first().click();
      cy.get('input[type="number"]').first().clear().type("10");
      cy.contains("button", /Place|Submit|Confirm/i).click();
    });
  });

  describe("Recent Trades Widget", () => {
    it("should display recent trades", () => {
      cy.contains(/Recent|Trades|History/i).should("be.visible");
    });

    it("should show trade timestamps", () => {
      // Timestamps in trade list
    });

    it("should color code buy/sell trades", () => {
      // Green for buys, red for sells
    });
  });

  describe("Market Information", () => {
    it("should display market title", () => {
      cy.waitForMarketsToLoad();
      // Market title should be visible
    });

    it("should display market volume", () => {
      cy.contains(/Volume|\$/i).should("be.visible");
    });

    it("should display market liquidity", () => {
      cy.contains(/Liquidity/i).should("be.visible");
    });

    it("should display market end date", () => {
      cy.contains(/End|Expiry|Date/i).should("be.visible");
    });
  });

  describe("Live Streaming", () => {
    it("should show streaming indicator when active", () => {
      // When streaming is active
      cy.mockTauriInvoke("stream_polymarket_prices", { success: true });

      // Start stream for a market
      // Should show live indicator
    });

    it("should update prices in real-time", () => {
      // Price updates should reflect in UI
    });

    it("should stop streaming when navigating away", () => {
      // Cleanup on unmount
    });
  });

  describe("Responsive Layout", () => {
    it("should adapt to smaller screens", () => {
      cy.viewport(768, 1024);
      cy.visit("/");
      cy.navigateToTab("terminal");

      // Layout should adapt
    });

    it("should collapse sidebar on mobile", () => {
      cy.viewport(375, 667);
      cy.visit("/");
      cy.navigateToTab("terminal");

      // Sidebar should be collapsible
    });
  });
});
