/// <reference types="cypress" />

describe("Scraper Tab", () => {
  beforeEach(() => {
    cy.visit("/");
    cy.navigateToTab("scraper");
  });

  describe("Layout", () => {
    it("should display scraper tab content", () => {
      cy.contains("button", "Scraper").should("have.class", "text-white");
    });

    it("should display market input section", () => {
      cy.get('input[placeholder*="polymarket"]').should("exist");
    });

    it("should display Find Market button", () => {
      cy.contains("button", "Find Market").should("be.visible");
    });
  });

  describe("Market Resolution", () => {
    it("should show error when input is empty", () => {
      cy.contains("button", "Find Market").click();
      cy.contains("Please enter a URL or Slug").should("be.visible");
    });

    it("should resolve market from URL", () => {
      const mockMetadata = {
        title: "Will Bitcoin reach $100k?",
        outcomes: [
          { id: "token-yes", name: "Yes" },
          { id: "token-no", name: "No" },
        ],
      };

      cy.mockTauriInvoke("resolve_polymarket_id", mockMetadata);

      cy.get('input[placeholder*="polymarket"]').type(
        "https://polymarket.com/event/btc-100k"
      );
      cy.contains("button", "Find Market").click();

      cy.contains("Resolving market").should("be.visible");
      cy.contains("Will Bitcoin reach $100k?", { timeout: 5000 }).should(
        "be.visible"
      );
    });

    it("should display outcomes when market is resolved", () => {
      const mockMetadata = {
        title: "Test Market",
        outcomes: [
          { id: "token-yes", name: "Yes" },
          { id: "token-no", name: "No" },
        ],
      };

      cy.mockTauriInvoke("resolve_polymarket_id", mockMetadata);

      cy.get('input[placeholder*="polymarket"]').type("test-market");
      cy.contains("button", "Find Market").click();

      cy.contains("Yes", { timeout: 5000 }).should("be.visible");
      cy.contains("No").should("be.visible");
    });

    it("should handle resolution error", () => {
      cy.window().then((win) => {
        const invoke = (win as any).__TAURI__?.core?.invoke;
        if (invoke && invoke.withArgs) {
          invoke.withArgs("resolve_polymarket_id").rejects(new Error("Market not found"));
        }
      });

      cy.get('input[placeholder*="polymarket"]').type("invalid-market");
      cy.contains("button", "Find Market").click();

      cy.contains("Error", { timeout: 5000 }).should("be.visible");
    });
  });

  describe("Outcome Selection", () => {
    beforeEach(() => {
      const mockMetadata = {
        title: "Test Market",
        outcomes: [
          { id: "token-1", name: "Option A" },
          { id: "token-2", name: "Option B" },
          { id: "token-3", name: "Option C" },
        ],
      };

      cy.mockTauriInvoke("resolve_polymarket_id", mockMetadata);

      cy.get('input[placeholder*="polymarket"]').type("test-market");
      cy.contains("button", "Find Market").click();
      cy.contains("Option A", { timeout: 5000 }).should("be.visible");
    });

    it("should allow selecting outcomes", () => {
      // Outcomes should be selectable via checkboxes
      cy.get('input[type="checkbox"]').should("have.length.at.least", 1);
    });

    it("should toggle outcome selection", () => {
      cy.get('input[type="checkbox"]').first().click();
      cy.get('input[type="checkbox"]').first().should("not.be.checked");

      cy.get('input[type="checkbox"]').first().click();
      cy.get('input[type="checkbox"]').first().should("be.checked");
    });
  });

  describe("Download Configuration", () => {
    beforeEach(() => {
      const mockMetadata = {
        title: "Test Market",
        outcomes: [
          { id: "token-yes", name: "Yes" },
          { id: "token-no", name: "No" },
        ],
      };

      cy.mockTauriInvoke("resolve_polymarket_id", mockMetadata);

      cy.get('input[placeholder*="polymarket"]').type("test-market");
      cy.contains("button", "Find Market").click();
      cy.contains("Yes", { timeout: 5000 }).should("be.visible");
    });

    it("should have frequency selector", () => {
      cy.contains("Frequency").should("be.visible");
      cy.get("select").should("exist");
    });

    it("should have date range inputs", () => {
      cy.get('input[type="date"]').should("have.length.at.least", 2);
    });

    it("should have download button", () => {
      cy.contains("button", "Download").should("exist");
    });
  });

  describe("Live Streaming", () => {
    beforeEach(() => {
      const mockMetadata = {
        title: "Test Market",
        outcomes: [
          { id: "token-yes", name: "Yes" },
          { id: "token-no", name: "No" },
        ],
      };

      cy.mockTauriInvoke("resolve_polymarket_id", mockMetadata);

      cy.get('input[placeholder*="polymarket"]').type("test-market");
      cy.contains("button", "Find Market").click();
      cy.contains("Yes", { timeout: 5000 }).should("be.visible");
    });

    it("should have stream controls", () => {
      cy.contains("Stream").should("exist");
    });

    it("should start streaming when Stream button is clicked", () => {
      cy.mockTauriInvoke("stream_polymarket_prices", { success: true });

      cy.contains("button", "Stream").click();

      // Should show streaming indicator or stop button
      cy.contains(/Stop|Streaming/i).should("be.visible");
    });
  });
});
