/// <reference types="cypress" />

describe("Dashboard", () => {
  beforeEach(() => {
    cy.visit("/");
    // Ensure we're on dashboard (default tab)
  });

  describe("Dashboard Layout", () => {
    it("should display the main dashboard sections", () => {
      // Market stats widget should be visible
      cy.get('[class*="MarketStats"], [class*="market-stats"]').should("exist");

      // Trending markets section should be visible
      cy.contains("Trending Markets").should("be.visible");
    });

    it("should display user profile widget on large screens", () => {
      // Set viewport to large screen
      cy.viewport(1400, 900);
      cy.visit("/");

      // Profile section should be visible on XL screens
      cy.get('[class*="UserProfile"]').should("exist");
    });

    it("should display global activity widget on large screens", () => {
      cy.viewport(1200, 900);
      cy.visit("/");

      cy.get('[class*="GlobalActivity"]').should("exist");
    });
  });

  describe("Trending Markets Widget", () => {
    it("should display loading state initially", () => {
      // On initial load, should show loading spinner
      cy.get('[class*="animate-spin"]').should("exist");
    });

    it("should display trending markets after loading", () => {
      // Wait for loading to complete
      cy.contains("Trending Markets").should("be.visible");

      // After mock data loads, should display market rows
      cy.waitForMarketsToLoad();

      // Should have market items
      cy.get("table tbody tr").should("have.length.at.least", 1);
    });

    it("should display market information in table", () => {
      cy.waitForMarketsToLoad();

      // Check table headers
      cy.contains("th", "Market").should("be.visible");
      cy.contains("th", "Chart").should("be.visible");
      cy.contains("th", "Price").should("be.visible");
      cy.contains("th", "Volume").should("be.visible");
      cy.contains("th", "Outcomes").should("be.visible");
    });

    it("should have clickable market rows", () => {
      cy.waitForMarketsToLoad();

      // First row should be clickable
      cy.get("table tbody tr").first().should("have.class", "cursor-pointer");
    });

    it("should navigate to terminal when market is clicked", () => {
      cy.waitForMarketsToLoad();

      // Click on first market row
      cy.get("table tbody tr").first().click();

      // Should navigate to Markets/Terminal tab
      cy.contains("button", "Markets").should("have.class", "text-white");
    });

    it("should have refresh button", () => {
      cy.contains("button", "Refresh").should("be.visible");
    });

    it("should refresh markets when refresh button is clicked", () => {
      cy.waitForMarketsToLoad();

      cy.contains("button", "Refresh").click();

      // Should show loading state again
      cy.get('[class*="animate-spin"]').should("exist");
    });

    it("should display market icons", () => {
      cy.waitForMarketsToLoad();

      // Markets should have icons (emoji or icon containers)
      cy.get("table tbody tr").first().find('[class*="rounded-full"]').should("exist");
    });

    it("should display probability bars", () => {
      cy.waitForMarketsToLoad();

      // Should have Y/N probability indicators
      cy.get("table tbody tr").first().within(() => {
        cy.contains("Y").should("exist");
        cy.contains("N").should("exist");
      });
    });

    it("should display sparkline charts", () => {
      cy.waitForMarketsToLoad();

      // Should have SVG sparklines
      cy.get("table tbody tr").first().find("svg").should("exist");
    });
  });

  describe("Favorites Widget", () => {
    it("should not display favorites widget when no favorites exist", () => {
      cy.clearFavorites();
      cy.visit("/");

      // Favorites widget should not be visible when empty
      cy.contains("Your Favorites").should("not.exist");
    });

    it("should display favorites widget when favorites exist", () => {
      // Add a favorite to localStorage
      cy.window().then((win) => {
        const favorites = [
          {
            id: "market-1",
            symbol: "BTC100K",
            name: "Will Bitcoin reach $100k?",
            addedAt: Date.now(),
            marketData: {
              id: "market-1",
              outcomes: [
                { id: "token-1-yes", name: "Yes" },
                { id: "token-1-no", name: "No" },
              ],
            },
          },
        ];
        win.localStorage.setItem("nglab-favorites", JSON.stringify(favorites));
      });

      cy.visit("/");

      // Favorites widget should now be visible
      cy.contains("Your Favorites").should("be.visible");
    });

    it("should navigate to favorites tab via Manage Favorites", () => {
      // Add a favorite first
      cy.window().then((win) => {
        const favorites = [
          {
            id: "market-1",
            symbol: "BTC100K",
            name: "Will Bitcoin reach $100k?",
            addedAt: Date.now(),
          },
        ];
        win.localStorage.setItem("nglab-favorites", JSON.stringify(favorites));
      });

      cy.visit("/");
      cy.contains("Manage Favorites").click();

      // Should navigate to favorites tab
      cy.contains("button", "Favorites").should("have.class", "text-white");
    });
  });

  describe("Error Handling", () => {
    it("should display error message when API fails", () => {
      // Mock API failure
      cy.mockTauriInvoke("get_public_polymarket_markets", {
        success: false,
        message: "Network error: Unable to fetch markets",
        data: null,
      });

      cy.visit("/");

      // Should display error message
      cy.contains("Network error").should("be.visible");
      cy.contains("Try again").should("be.visible");
    });

    it("should retry loading when Try again is clicked", () => {
      // First request fails
      cy.mockTauriInvoke("get_public_polymarket_markets", {
        success: false,
        message: "Network error",
        data: null,
      });

      cy.visit("/");
      cy.contains("Try again").should("be.visible");

      // Mock successful response for retry
      cy.mockTauriInvoke("get_public_polymarket_markets", {
        success: true,
        message: "Found 5 markets",
        data: [
          {
            id: "market-1",
            event_id: "event-1",
            title: "Test Market",
            question: "Test?",
            outcomes: ["Yes", "No"],
            clob_token_ids: ["yes-1", "no-1"],
            volume: 1000000,
            liquidity: 50000,
            end_date: "2026-12-31",
            active: true,
          },
        ],
      });

      cy.contains("button", "Try again").click();

      // Should load markets
      cy.waitForMarketsToLoad();
    });
  });
});
