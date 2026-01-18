/// <reference types="cypress" />

describe("Favorites Tab", () => {
  beforeEach(() => {
    cy.clearFavorites();
    cy.visit("/");
    cy.navigateToTab("favorites");
  });

  describe("Favorites Tab Layout", () => {
    it("should display the favorites tab header", () => {
      cy.contains("Favorite Markets").should("be.visible");
      cy.contains("Search and add markets").should("be.visible");
    });

    it("should have two panels - search and favorites list", () => {
      // Left panel: Search
      cy.get('input[placeholder*="Search markets"]').should("be.visible");

      // Right panel: Your Favorites
      cy.contains("Your Favorites").should("be.visible");
    });

    it("should display empty state when no favorites", () => {
      cy.contains("No favorites yet").should("be.visible");
    });
  });

  describe("Market Search", () => {
    it("should display search input", () => {
      cy.get('input[placeholder*="Search markets"]')
        .should("be.visible")
        .should("have.attr", "type", "text");
    });

    it("should display open markets by default", () => {
      cy.waitForMarketsToLoad();
      cy.contains("Open Markets").should("be.visible");
    });

    it("should search markets when typing", () => {
      // Type in search
      cy.get('input[placeholder*="Search markets"]').type("Bitcoin");

      // Should show searching state
      cy.contains("Searching").should("be.visible");

      // Wait for search results
      cy.waitForMarketsToLoad();
    });

    it("should clear search when X button is clicked", () => {
      cy.get('input[placeholder*="Search markets"]').type("test");

      // Clear button should appear
      cy.get('input[placeholder*="Search markets"]')
        .parent()
        .find("button")
        .click();

      // Input should be cleared
      cy.get('input[placeholder*="Search markets"]').should("have.value", "");
    });

    it("should display no results message for unknown search", () => {
      cy.mockTauriInvoke("search_public_polymarket_markets", {
        success: true,
        message: "Found 0 markets",
        data: [],
      });

      cy.get('input[placeholder*="Search markets"]').type("xyznonexistent");

      cy.waitForMarketsToLoad();
      cy.contains("No markets found").should("be.visible");
    });

    it("should debounce search input", () => {
      // Type quickly
      cy.get('input[placeholder*="Search markets"]').type("test", {
        delay: 50,
      });

      // Search should be debounced (300ms delay)
      cy.wait(100);
      // Should not have started searching yet
      cy.contains("Searching").should("not.exist");

      // After debounce period
      cy.wait(300);
      // Now it should search
    });
  });

  describe("Adding Favorites", () => {
    it("should display star button for each market", () => {
      cy.waitForMarketsToLoad();

      // Each market item should have a star button
      cy.get('[class*="divide-y"] > div')
        .first()
        .find("button")
        .should("exist");
    });

    it("should add market to favorites when star is clicked", () => {
      cy.waitForMarketsToLoad();

      // Get the count before adding
      cy.contains("0 markets").should("be.visible");

      // Click star on first market
      cy.get('[class*="divide-y"] > div')
        .first()
        .find("button")
        .first()
        .click();

      // Count should update
      cy.contains("1 market").should("be.visible");
    });

    it("should show filled star for favorited markets", () => {
      cy.waitForMarketsToLoad();

      // Add to favorites
      cy.get('[class*="divide-y"] > div')
        .first()
        .find("button")
        .first()
        .click();

      // Star should be filled (has fill-current class or similar)
      cy.get('[class*="divide-y"] > div')
        .first()
        .find("button")
        .first()
        .should("have.class", "bg-yellow-500/20");
    });

    it("should persist favorites to localStorage", () => {
      cy.waitForMarketsToLoad();

      // Add to favorites
      cy.get('[class*="divide-y"] > div')
        .first()
        .find("button")
        .first()
        .click();

      // Check localStorage
      cy.window().then((win) => {
        const favorites = JSON.parse(
          win.localStorage.getItem("nglab-favorites") || "[]"
        );
        expect(favorites).to.have.length(1);
      });
    });

    it("should display favorites in right panel after adding", () => {
      cy.waitForMarketsToLoad();

      // Add to favorites
      cy.get('[class*="divide-y"] > div')
        .first()
        .find("button")
        .first()
        .click();

      // Should appear in Your Favorites panel
      cy.contains("Your Favorites")
        .parent()
        .parent()
        .within(() => {
          cy.get('[class*="divide-y"] > div').should("have.length", 1);
        });
    });
  });

  describe("Removing Favorites", () => {
    beforeEach(() => {
      // Pre-populate with a favorite
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
      cy.navigateToTab("favorites");
    });

    it("should display X button for favorites", () => {
      // Find the favorites panel and check for remove button
      cy.contains("Your Favorites")
        .parent()
        .parent()
        .find('button[title*="Remove"]')
        .should("exist");
    });

    it("should remove favorite when X is clicked", () => {
      cy.contains("1 market").should("be.visible");

      // Click remove button
      cy.contains("Your Favorites")
        .parent()
        .parent()
        .find('button[title*="Remove"]')
        .first()
        .click();

      // Should show empty state
      cy.contains("No favorites yet").should("be.visible");
    });

    it("should update localStorage when favorite is removed", () => {
      // Remove favorite
      cy.contains("Your Favorites")
        .parent()
        .parent()
        .find('button[title*="Remove"]')
        .first()
        .click();

      // Check localStorage
      cy.window().then((win) => {
        const favorites = JSON.parse(
          win.localStorage.getItem("nglab-favorites") || "[]"
        );
        expect(favorites).to.have.length(0);
      });
    });

    it("should toggle favorite off by clicking star again", () => {
      cy.waitForMarketsToLoad();

      // Find the favorited market in search results and click to unfavorite
      cy.contains("1 market").should("be.visible");

      // Click the filled star to remove
      cy.get('[class*="bg-yellow-500"]').first().click();

      // Should be removed
      cy.contains("0 markets").should("be.visible");
    });
  });

  describe("Navigate to Market", () => {
    beforeEach(() => {
      // Pre-populate with a favorite
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
      cy.navigateToTab("favorites");
    });

    it("should have navigate button for favorites", () => {
      cy.contains("Your Favorites")
        .parent()
        .parent()
        .find('button[title*="Open"]')
        .should("exist");
    });

    it("should navigate to terminal when navigate button is clicked", () => {
      cy.contains("Your Favorites")
        .parent()
        .parent()
        .find('button[title*="Open"]')
        .first()
        .click();

      // Should navigate to Markets tab
      cy.contains("button", "Markets").should("have.class", "text-white");
    });
  });

  describe("Favorites Badge in Navigation", () => {
    it("should not show badge when no favorites", () => {
      cy.clearFavorites();
      cy.visit("/");

      // Favorites button should not have a badge
      cy.contains("button", "Favorites").within(() => {
        cy.get("span").should("not.exist");
      });
    });

    it("should show badge with count when favorites exist", () => {
      cy.window().then((win) => {
        const favorites = [
          {
            id: "market-1",
            symbol: "BTC100K",
            name: "Test 1",
            addedAt: Date.now(),
          },
          {
            id: "market-2",
            symbol: "ETH10K",
            name: "Test 2",
            addedAt: Date.now(),
          },
        ];
        win.localStorage.setItem("nglab-favorites", JSON.stringify(favorites));
      });

      cy.visit("/");

      // Favorites button should have badge with count
      cy.contains("button", "Favorites").within(() => {
        cy.contains("2").should("be.visible");
      });
    });
  });

  describe("Error Handling", () => {
    it("should display error when markets fail to load", () => {
      cy.mockTauriInvoke("get_public_polymarket_markets", {
        success: false,
        message: "Failed to connect to Polymarket",
        data: null,
      });

      cy.visit("/");
      cy.navigateToTab("favorites");

      cy.contains("Failed to connect").should("be.visible");
      cy.contains("Try again").should("be.visible");
    });

    it("should retry loading when Try again is clicked", () => {
      cy.mockTauriInvoke("get_public_polymarket_markets", {
        success: false,
        message: "Network error",
        data: null,
      });

      cy.visit("/");
      cy.navigateToTab("favorites");

      cy.contains("Try again").click();

      // Should attempt to reload
      cy.get('[class*="animate-spin"]').should("exist");
    });
  });
});
