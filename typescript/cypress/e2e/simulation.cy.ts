/// <reference types="cypress" />

describe("Simulation Tab", () => {
  beforeEach(() => {
    cy.visit("/");
    cy.navigateToTab("simulation");
  });

  describe("Layout", () => {
    it("should display simulation tab content", () => {
      cy.contains("button", "Simulation").should("have.class", "text-white");
    });

    it("should display price chart section", () => {
      cy.contains("Live Price").should("be.visible");
    });

    it("should display order book depth section", () => {
      cy.contains("Order Book Depth").should("be.visible");
    });

    it("should display current step stats", () => {
      cy.contains("Current Step").should("be.visible");
    });

    it("should display portfolio value stats", () => {
      cy.contains("Portfolio Value").should("be.visible");
    });

    it("should display agent logs section", () => {
      cy.contains("Agent Logs").should("be.visible");
    });
  });

  describe("Simulation Controls", () => {
    it("should display Start button when simulation is not running", () => {
      cy.get("header").within(() => {
        cy.contains("button", "Start").should("be.visible");
      });
    });

    it("should have a reset button", () => {
      cy.get("header").within(() => {
        cy.get('button svg[class*="lucide-rotate"]').parent().should("exist");
      });
    });

    it("should start simulation when Start is clicked", () => {
      cy.mockTauriInvoke("start_simulation", { success: true });

      cy.get("header").within(() => {
        cy.contains("button", "Start").click();
      });

      // Button should change to Stop
      cy.get("header").within(() => {
        cy.contains("button", "Stop").should("be.visible");
      });
    });

    it("should stop simulation when Stop is clicked", () => {
      cy.mockTauriInvoke("start_simulation", { success: true });
      cy.mockTauriInvoke("stop_simulation", { success: true });

      // Start first
      cy.get("header").within(() => {
        cy.contains("button", "Start").click();
      });

      // Then stop
      cy.get("header").within(() => {
        cy.contains("button", "Stop").click();
      });

      // Should show Start again
      cy.get("header").within(() => {
        cy.contains("button", "Start").should("be.visible");
      });
    });
  });

  describe("Price Chart", () => {
    it("should display chart container", () => {
      cy.get('[class*="PriceChart"], canvas').should("exist");
    });

    it("should show live price indicators when streaming", () => {
      // When streaming is active, should show price badges
      cy.contains("Live Price").should("be.visible");
    });
  });

  describe("Order Book", () => {
    it("should display order book visualization", () => {
      cy.contains("Order Book Depth").should("be.visible");
    });
  });

  describe("Stats Panel", () => {
    it("should display initial portfolio value", () => {
      cy.contains("10000.00").should("be.visible");
    });

    it("should display step counter", () => {
      cy.contains("Current Step").parent().within(() => {
        cy.get("p").last().should("contain.text", "0");
      });
    });
  });

  describe("Agent Logs", () => {
    it("should display waiting message when no logs", () => {
      cy.contains("Waiting for simulation data").should("be.visible");
    });
  });

  describe("Simulation State Persistence", () => {
    it("should not show controls in other tabs", () => {
      cy.navigateToTab("dashboard");

      cy.get("header").within(() => {
        cy.contains("button", "Start").should("not.exist");
      });
    });

    it("should show controls when returning to simulation tab", () => {
      cy.navigateToTab("dashboard");
      cy.navigateToTab("simulation");

      cy.get("header").within(() => {
        cy.contains("button", "Start").should("be.visible");
      });
    });
  });
});
