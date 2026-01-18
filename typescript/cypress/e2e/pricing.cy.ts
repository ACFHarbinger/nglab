/// <reference types="cypress" />

describe("Pricing Tab", () => {
  beforeEach(() => {
    cy.visit("/");
    cy.navigateToTab("pricing");
  });

  describe("Layout", () => {
    it("should display pricing tab content", () => {
      cy.contains("button", "Pricing").should("have.class", "text-white");
    });

    it("should display model selector", () => {
      cy.contains("Black-Scholes").should("be.visible");
    });

    it("should display pricing form inputs", () => {
      cy.contains("Spot Price").should("be.visible");
      cy.contains("Strike Price").should("be.visible");
    });
  });

  describe("Model Selection", () => {
    it("should have Black-Scholes selected by default", () => {
      cy.get('button').contains("Black-Scholes").should("have.class", "bg-indigo-600");
    });

    it("should switch to Rough Bergomi model", () => {
      cy.contains("button", "Rough Bergomi").click();
      cy.contains("button", "Rough Bergomi").should("have.class", "bg-indigo-600");
    });

    it("should switch to Rough Heston model", () => {
      cy.contains("button", "Rough Heston").click();
      cy.contains("button", "Rough Heston").should("have.class", "bg-indigo-600");
    });

    it("should switch to Credit Risk model", () => {
      cy.contains("button", "Credit Risk").click();
      cy.contains("button", "Credit Risk").should("have.class", "bg-indigo-600");
    });
  });

  describe("Black-Scholes Model", () => {
    it("should display BSM-specific inputs", () => {
      cy.contains("Volatility").should("be.visible");
    });

    it("should display option type selector", () => {
      cy.contains("Call").should("be.visible");
      cy.contains("Put").should("be.visible");
    });

    it("should calculate option price", () => {
      const mockResult = {
        call: 10.45,
        put: 5.67,
        delta: 0.65,
        gamma: 0.02,
        vega: 25.3,
        d1: 0.5,
        d2: 0.3,
      };

      cy.mockTauriInvoke("pricing_black_scholes", mockResult);

      cy.contains("button", "Calculate").click();

      cy.contains("Call Price", { timeout: 5000 }).should("be.visible");
    });

    it("should display Greeks after calculation", () => {
      const mockResult = {
        call: 10.45,
        put: 5.67,
        delta: 0.65,
        gamma: 0.02,
        vega: 25.3,
        d1: 0.5,
        d2: 0.3,
      };

      cy.mockTauriInvoke("pricing_black_scholes", mockResult);

      cy.contains("button", "Calculate").click();

      cy.contains("Delta", { timeout: 5000 }).should("be.visible");
      cy.contains("Gamma").should("be.visible");
      cy.contains("Vega").should("be.visible");
    });

    it("should update inputs", () => {
      cy.get('input').first().clear().type("150");
      cy.get('input').first().should("have.value", "150");
    });
  });

  describe("Rough Bergomi Model", () => {
    beforeEach(() => {
      cy.contains("button", "Rough Bergomi").click();
    });

    it("should display rBergomi-specific inputs", () => {
      cy.contains("Xi").should("be.visible");
      cy.contains("Eta").should("be.visible");
      cy.contains("Hurst").should("be.visible");
      cy.contains("Rho").should("be.visible");
    });

    it("should display Monte Carlo parameters", () => {
      cy.contains("Paths").should("be.visible");
      cy.contains("Steps").should("be.visible");
    });

    it("should calculate rBergomi price", () => {
      const mockResult = {
        price: 12.34,
        std_error: 0.05,
        mean_terminal: 105.5,
      };

      cy.mockTauriInvoke("pricing_rbergomi", mockResult);

      cy.contains("button", "Calculate").click();

      cy.contains("Price", { timeout: 10000 }).should("be.visible");
    });

    it("should display standard error", () => {
      const mockResult = {
        price: 12.34,
        std_error: 0.05,
        mean_terminal: 105.5,
      };

      cy.mockTauriInvoke("pricing_rbergomi", mockResult);

      cy.contains("button", "Calculate").click();

      cy.contains("Std Error", { timeout: 10000 }).should("be.visible");
    });
  });

  describe("Rough Heston Model", () => {
    beforeEach(() => {
      cy.contains("button", "Rough Heston").click();
    });

    it("should display Heston-specific inputs", () => {
      cy.contains("V0").should("be.visible");
      cy.contains("Theta").should("be.visible");
      cy.contains("Kappa").should("be.visible");
    });

    it("should calculate Rough Heston price", () => {
      const mockResult = {
        price: 11.23,
        std_error: 0.03,
        mean_terminal: 102.5,
        p05: 80,
        p95: 130,
        paths: 10000,
        steps: 100,
      };

      cy.mockTauriInvoke("pricing_rough_heston", mockResult);

      cy.contains("button", "Calculate").click();

      cy.contains("Price", { timeout: 10000 }).should("be.visible");
    });
  });

  describe("Credit Risk Model", () => {
    beforeEach(() => {
      cy.contains("button", "Credit Risk").click();
    });

    it("should display credit-specific inputs", () => {
      cy.contains("Default").should("be.visible");
      cy.contains("Recovery").should("be.visible");
    });

    it("should calculate CVA", () => {
      const mockResult = {
        base: 100,
        adjusted: 95.5,
        survival: 0.98,
        cva: 4.5,
      };

      cy.mockTauriInvoke("pricing_credit_risk", mockResult);

      cy.contains("button", "Calculate").click();

      cy.contains("CVA", { timeout: 5000 }).should("be.visible");
    });

    it("should display survival probability", () => {
      const mockResult = {
        base: 100,
        adjusted: 95.5,
        survival: 0.98,
        cva: 4.5,
      };

      cy.mockTauriInvoke("pricing_credit_risk", mockResult);

      cy.contains("button", "Calculate").click();

      cy.contains("Survival", { timeout: 5000 }).should("be.visible");
    });
  });

  describe("Input Validation", () => {
    it("should handle negative inputs gracefully", () => {
      cy.get('input').first().clear().type("-100");
      cy.contains("button", "Calculate").click();
      // Should either clamp or show validation error
    });

    it("should handle zero values", () => {
      cy.get('input').first().clear().type("0");
      cy.contains("button", "Calculate").click();
    });
  });

  describe("Loading State", () => {
    it("should show loading indicator during calculation", () => {
      cy.mockTauriInvoke("pricing_black_scholes", {
        call: 10,
        put: 5,
        delta: 0.5,
        gamma: 0.02,
        vega: 20,
        d1: 0.5,
        d2: 0.3,
      });

      cy.contains("button", "Calculate").click();

      // Should show loading state
      cy.get('[class*="animate-spin"]').should("exist");
    });
  });
});
