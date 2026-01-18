/// <reference types="cypress" />

// Custom commands for NGLab testing

declare global {
  namespace Cypress {
    interface Chainable {
      /**
       * Mock Tauri invoke with custom response
       */
      mockTauriInvoke(command: string, response: any): Chainable<void>;

      /**
       * Navigate to a specific tab
       */
      navigateToTab(
        tabName:
          | "dashboard"
          | "prediction"
          | "terminal"
          | "analysis"
          | "favorites"
          | "simulation"
          | "scraper"
          | "pricing"
          | "training"
          | "news"
          | "vault"
          | "account"
      ): Chainable<void>;

      /**
       * Wait for markets to load
       */
      waitForMarketsToLoad(): Chainable<void>;

      /**
       * Add a market to favorites
       */
      addToFavorites(marketId: string): Chainable<void>;

      /**
       * Remove a market from favorites
       */
      removeFromFavorites(marketId: string): Chainable<void>;

      /**
       * Clear all favorites from localStorage
       */
      clearFavorites(): Chainable<void>;
    }
  }
}

// Mock Tauri invoke with custom response
Cypress.Commands.add("mockTauriInvoke", (command: string, response: any) => {
  cy.window().then((win) => {
    const invoke = (win as any).__TAURI__?.core?.invoke;
    if (invoke && invoke.withArgs) {
      invoke.withArgs(command).resolves(response);
    }
  });
});

// Navigate to a specific tab
Cypress.Commands.add("navigateToTab", (tabName: string) => {
  const tabLabels: Record<string, string> = {
    dashboard: "Dashboard",
    prediction: "Intelligence",
    terminal: "Markets",
    analysis: "Positions",
    favorites: "Favorites",
    simulation: "Simulation",
    scraper: "Scraper",
    pricing: "Pricing",
    training: "Training",
    news: "News",
    vault: "Vault",
    account: "Account",
  };

  const label = tabLabels[tabName];
  if (label) {
    cy.contains("button", label).click();
  }
});

// Wait for markets to load (spinner disappears)
Cypress.Commands.add("waitForMarketsToLoad", () => {
  // Wait for loading spinner to disappear
  cy.get('[class*="animate-spin"]', { timeout: 10000 }).should("not.exist");
});

// Add a market to favorites by clicking its star button
Cypress.Commands.add("addToFavorites", (marketId: string) => {
  cy.get(`[data-market-id="${marketId}"]`)
    .find('button[title*="favorite"], button:has(svg)')
    .first()
    .click();
});

// Remove a market from favorites
Cypress.Commands.add("removeFromFavorites", (marketId: string) => {
  cy.get(`[data-market-id="${marketId}"]`)
    .find('button[title*="Remove"], button:has(svg)')
    .first()
    .click();
});

// Clear favorites from localStorage
Cypress.Commands.add("clearFavorites", () => {
  cy.window().then((win) => {
    win.localStorage.removeItem("nglab-favorites");
  });
});

export {};
